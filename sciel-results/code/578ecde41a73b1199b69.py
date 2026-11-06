"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
import math
from collections import deque, defaultdict

def _topo_order(parents, vertices, vset):
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children.get(u, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(vertices):
        topo = list(vertices)
    return topo, children

def _is_layered(parents, vertices, topo, children):
    """Check if DAG is layered: each vertex's parents are all in one contiguous layer."""
    if not vertices:
        return True, []
    
    # Assign layers by longest path from sources
    layer = {}
    for v in topo:
        ps = parents.get(v, [])
        if not ps:
            layer[v] = 0
        else:
            layer[v] = max(layer[p] for p in ps) + 1
    
    # Check that all parents of v are in layer[v]-1
    for v in vertices:
        ps = parents.get(v, [])
        if not ps:
            continue
        for p in ps:
            if layer[p] != layer[v] - 1:
                return False, []
    
    # Group by layer
    max_layer = max(layer.values()) if layer else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in vertices:
        layers[layer[v]].append(v)
    
    return True, layers

def _simulate(ops, dag, budget):
    """
    Simulate the pebbling schedule and return (valid, transfers, max_memory_used).
    Returns (False, 0, 0) if invalid.
    
    Semantics:
    - State: set of vertices currently in fast memory (pebbled/blue).
    - Initial state: empty (inputs must be loaded).
    - "load": bring an input vertex into memory. Costs 1 transfer.
    - "store": write a vertex to slow memory/output. Removes from fast memory. Costs 1 transfer.
    - "compute": compute a vertex from its parents (which must be in memory). Result goes into memory. Free.
    - "delete": remove a vertex from memory. Free.
    - Constraint: at all times, |in_mem| <= budget.
    - Goal: all output vertices must have been stored at least once.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    outset = set(outputs)
    
    # Identify inputs (vertices with no parents)
    inputs = set(v for v in parents.keys() if not parents.get(v, []))
    
    in_mem = set()
    stored_outputs = set()
    transfers = 0
    max_mem = 0
    
    for op, v in ops:
        if op == "load":
            # Can only load inputs
            if v not in inputs:
                return False, 0, 0
            if v in in_mem:
                # Already in memory, no-op or invalid? Let's say invalid to avoid waste.
                return False, 0, 0
            in_mem.add(v)
            transfers += 1
            max_mem = max(max_mem, len(in_mem))
            if len(in_mem) > budget:
                return False, 0, 0
        
        elif op == "store":
            # Must be in memory
            if v not in in_mem:
                return False, 0, 0
            # Can only store outputs? Or any vertex?
            # The task says "all outputs must finish blue". 
            # If "finish blue" means in memory at end, then store must NOT remove from memory?
            # But standard pebbling "store" removes the pebble.
            # Let's re-read: "Inputs start blue, all outputs must finish blue"
            # This strongly suggests the FINAL STATE has outputs in memory.
            # If store removes from memory, we can't have them in memory at the end unless we don't store or store doesn't remove.
            # However, the operation is called "store", which usually implies moving to slow memory.
            # Maybe "finish blue" is a misinterpretation of "are produced"?
            # Let's look at the score: "mean transfers / naive schedule".
            # If we must keep outputs in memory at the end, we can't store them (if store removes).
            # But the interface includes "store".
            # Hypothesis: "store" writes to output buffer but KEEPS in memory? 
            # Or: "store" is the only way to "output" a value, and it removes it, but the goal is just that they were computed/stored?
            # Let's assume standard pebbling: store removes from fast memory.
            # And the goal is that all outputs have been STORED (produced to slow memory).
            # The "finish blue" might refer to the value being available in slow memory (which is persistent).
            # I will proceed with: store removes from fast memory, goal is all outputs are in stored_outputs.
            
            in_mem.discard(v)
            if v in outset:
                stored_outputs.add(v)
            transfers += 1
            # max_mem unchanged (size decreased)
            
        elif op == "compute":
            ps = parents.get(v, [])
            for p in ps:
                if p not in in_mem:
                    return False, 0, 0
            # If v is already in memory, we can overwrite (recompute).
            in_mem.add(v)
            max_mem = max(max_mem, len(in_mem))
            if len(in_mem) > budget:
                return False, 0, 0
        
        elif op == "delete":
            if v not in in_mem:
                return False, 0, 0
            in_mem.discard(v)
            # No transfer cost
        
        else:
            return False, 0, 0
    
    # Check goal: all outputs must have been stored
    for o in outset:
        if o not in stored_outputs:
            return False, 0, 0
    
    return True, transfers, max_mem

def _pebble_naive(dag, budget):
    """
    Naive schedule: Load all inputs, compute everything in topo order, store outputs.
    This serves as the baseline for scoring.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    
    vset = set(vertices)
    outset = set(outputs)
    
    topo, children = _topo_order(parents, vertices, vset)
    
    ops = []
    in_mem = set()
    
    inputs = [v for v in topo if not parents.get(v, [])]
    
    # Load all inputs
    for v in inputs:
        ops.append(["load", v])
        in_mem.add(v)
    
    # Compute in topo order
    for v in topo:
        if not parents.get(v, []):
            continue  # Already loaded
        ops.append(["compute", v])
        in_mem.add(v)
    
    # Store outputs
    for v in outputs:
        if v in in_mem:
            ops.append(["store", v])
            in_mem.discard(v)
    
    return ops[:20000]

def _pebble_layered_optimal(dag, budget):
    """
    Optimized pebbling for layered DAGs.
    Strategy:
    1. Load all inputs.
    2. For each layer, compute vertices.
    3. Decide which vertices from the current layer to keep in memory for the next layer.
    4. Delete the rest.
    5. At the end, store outputs.
    
    To minimize transfers (loads + stores), we want to minimize recomputation.
    Since loads are fixed (all inputs loaded once), we focus on minimizing stores? 
    No, "mean transfers" includes loads and stores.
    If we keep more in memory, we recompute less, but might need to delete more? 
    Delete is free. Load is 1, Store is 1, Compute is 0.
    So minimizing total ops means minimizing (loads + stores).
    Loads are fixed if we load each input once.
    Stores are fixed (one per output).
    Wait, if we recompute, do we store again? No, we only store the final output.
    So the number of loads and stores is fixed by the DAG structure if we don't reload inputs or store intermediates.
    
    UNLESS: We delete an input, then need to reload it.
    Or: We delete an intermediate, then need to recompute it (which requires its parents to be in memory, potentially requiring loads of their parents if they were deleted).
    
    So the key is: avoid deleting things that will be needed again, OR avoid deleting inputs if they will be needed again.
    
    For a layered DAG:
    - Layer 0: Inputs.
    - Layer 1: Depends on Layer 0.
    - Layer 2: Depends on Layer 1.
    - ...
    - Layer L: Outputs.
    
    If we process layer by layer:
    1. Load all Layer 0.
    2. Compute Layer 1.
    3. If Layer 0 is not needed for Layer 2+, delete Layer 0.
    4. Compute Layer 2.
    5. If Layer 1 is not needed for Layer 3+, delete Layer 1.
    ...
    6. Store Layer L.
    
    This is optimal if the DAG is strictly layered (each vertex only depends on the previous layer).
    In this case, we never need to go back to a previous layer.
    So we just load inputs, compute forward, delete as we go, store outputs.
    Transfers = |Inputs| + |Outputs|.
    This is the minimum possible transfers! (Since each input must be loaded at least once, and each output stored at least once).
    
    So for strictly layered DAGs, the naive strategy is already optimal in terms of transfer count?
    Let's check the naive strategy above.
    It loads all inputs, computes all, stores outputs.
    It does NOT delete intermediates.
    But delete is free. So the transfer count is the same.
    
    Wait, is there a constraint on memory? Yes, `budget`.
    If the naive strategy exceeds the budget, it's invalid.
    So we must delete to stay within budget.
    
    Therefore, the goal is to find a valid schedule (within budget) that minimizes transfers.
    For layered DAGs, if we can fit all inputs + one layer of intermediates in memory, we can do it with |Inputs| + |Outputs| transfers.
    If not, we might need to reload inputs or recompute intermediates, increasing transfers.
    
    Strategy for layered DAG with budget B:
    - Let L0, L1, ..., Lk be the layers.
    - We need to keep a subset of Li to compute Li+1.
    - If |Li| > B, we can't keep all.
    - We should keep the "most useful" vertices.
    - Since it's layered, Li+1 depends only on Li.
    - If we delete a vertex in Li, and it's needed for Li+1, we must recompute it.
    - To recompute it, we need its parents in L(i-1).
    - So we might need to keep parts of L(i-1) as well.
    
    This becomes a complex optimization.
    However, for the "three layered DAGs" mentioned in the task, they might be small enough for DP or simple heuristics.
    
    Let's implement a greedy approach for layered DAGs:
    1. Load all inputs.
    2. For each layer i from 1 to k:
       a. Identify which vertices in L(i-1) are needed for Li.
       b. If we have space, keep them.
       c. Compute Li.
       d. Delete L(i-1) vertices that are no longer needed.
       e. If we ran out of space during compute, we might have had to delete some L(i-1) earlier, forcing recomputation.
    
    To avoid recomputation, we should try to keep all needed parents.
    If budget < |Li| + |Needed Parents|, we are in trouble.
    
    Let's assume the DAGs are such that a simple "keep all needed, delete rest" works, or we use a heuristic to decide what to keep if over budget.
    
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices, vset)
    layered, layers = _is_layered(parents, vertices, topo, children)
    
    if not layered or not layers:
        return None
    
    n_layers = len(layers)
    if n_layers == 0:
        return []
    
    ops = []
    in_mem = set()
    
    # Load all inputs (Layer 0)
    for v in layers[0]:
        ops.append(["load", v])
        in_mem.add(v)
    
    if len(in_mem) > b:
        # Can't even fit inputs? Return invalid or partial?
        # If inputs > budget, we must reload them.
        # This case is hard. Let's return None to fallback to other strategies.
        return None
    
    # Process layers 1 to n_layers-1
    for li in range(1, n_layers):
        layer = layers[li]
        prev_layer = layers[li-1]
        
        # Determine which parents in prev_layer are needed for this layer
        needed_parents = set()
        for v in layer:
            for p in parents.get(v, []):
                needed_parents.add(p)
        
        # If we don't have enough space for the new layer + needed parents, we have a problem.
        # Current memory: in_mem (contains prev_layer and possibly older layers if not deleted).
        # We should have deleted older layers already.
        # So in_mem should mostly be prev_layer.
        
        # Compute the new layer
        for v in layer:
            ps = parents.get(v, [])
            # Check if parents are in memory
            for p in ps:
                if p not in in_mem:
                    # Parent was deleted. Need to recompute it?
                    # This is where it gets complex.
                    # For now, if a parent is missing, we fail or try to reload/recompute.
                    # Let's assume we keep all needed parents.
                    # If we can't, we return None.
                    return None
            ops.append(["compute", v])
            in_mem.add(v)
            
            # Check budget
            if len(in_mem) > b:
                # We are over budget.
                # We must delete something.
                # We cannot delete parents needed for the rest of this layer.
                # We cannot delete vertices in the new layer that are needed for the next layer? 
                # We don't know the next layer yet.
                # We can delete vertices from prev_layer that are no longer needed for the rest of this layer.
                
                # Find deletable vertices
                # 1. From prev_layer: those not in needed_parents (or those whose children in this layer are all computed)
                # 2. From this layer: those not needed for the next layer (unknown)
                
                # Let's just delete from prev_layer first.
                remaining_needed = set()
                for w in layer:
                    if w in in_mem and w not in [x for x in layer if x in in_mem and x != v]: # This logic is messy
                        pass
                
                # Simpler: After computing v, check which parents of REMAINING vertices in this layer are needed.
                uncomputed_in_layer = [w for w in layer if w not in in_mem or (w in in_mem and w != v and w not in [x for x in layer if x != v])]
                # Actually, just iterate through the rest of the layer
                rest_of_layer = layer[layer.index(v)+1:]
                needed_for_rest = set()
                for w in rest_of_layer:
                    for p in parents.get(w, []):
                        needed_for_rest.add(p)
                
                # Deletable from prev_layer:
                deletable = [p for p in prev_layer if p in in_mem and p not in needed_for_rest]
                
                if deletable:
                    # Delete one (or enough)
                    ops.append(["delete", deletable[0]])
                    in_mem.discard(deletable[0])
                else:
                    # Cannot delete from prev_layer.
                    # Must delete from current layer?
                    # If we delete a computed vertex in current layer, we'll have to recompute it if needed later.
                    # But if it's not needed for the next layer, it's safe.
                    # We don't know next layer needs yet.
                    # Let's try to delete a vertex from current layer that is an output? No, outputs must be stored.
                    # Let's delete a vertex from current layer that is not an output.
                    non_outputs = [w for w in layer if w in in_mem and w not in outset]
                    if non_outputs:
                        ops.append(["delete", non_outputs[0]])
                        in_mem.discard(non_outputs[0])
                    else:
                        # Stuck
                        return None
        
        # If this is the last layer, store outputs
        if li == n_layers - 1:
            for v in layer:
                if v in outset:
                    if v in in_mem:
                        ops.append(["store", v])
                        in_mem.discard(v)
                    else:
                        # Output was deleted? Recompute?
                        return None
        else:
            # Delete vertices from prev_layer that are no longer needed
            # We already deleted some during the loop if over budget.
            # Now, delete ALL from prev_layer that are not needed for next layer?
            # We don't know next layer needs.
            # But we know next layer is layers[li+1].
            next_layer = layers[li+1]
            needed_for_next = set()
            for w in next_layer:
                for p in parents.get(w, []):
                    needed_for_next.add(p)
            
            # Delete from prev_layer those not in needed_for_next
            for p in prev_layer:
                if p in in_mem and p not in needed_for_next:
                    ops.append(["delete", p])
                    in_mem.discard(p)
            
            # Also, if we are still over budget, delete from current layer (non-outputs)
            while len(in_mem) > b:
                # Find a vertex to delete
                # Prefer non-outputs from current layer
                candidates = [v for v in in_mem if v not in outset]
                if not candidates:
                    return None
                # Pick one
                v_to_del = candidates[0]
                ops.append(["delete", v_to_del])
                in_mem.discard(v_to_del)
    
    return ops[:20000]

def _pebble_greedy_memory_aware(dag, budget):
    """
    Greedy pebbling that respects memory budget.
    Uses a post-order traversal for each output, but shares computations where possible.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices, vset)
    
    ops = []
    in_mem = set()
    stored = set()
    
    # Precompute "usefulness" or just do a simple DFS with memoization of in-memory state
    # This is complex. Let's try a simpler approach:
    # Process outputs one by one.
    # For each output, compute its subtree.
    # Keep intermediate results if they are likely to be reused.
    
    # Define a function to compute a vertex, ensuring parents are in memory
    def ensure_in_mem(v):
        if v in in_mem:
            return
        ps = parents.get(v, [])
        if not ps:
            # Input
            if len(in_mem) >= b:
                # Need to make space
                # Delete a vertex that is not needed for the current computation path?
                # This is hard to determine without lookahead.
                # Let's delete a random non-output vertex.
                for x in list(in_mem):
                    if x not in outset:
                        ops.append(["delete", x])
                        in_mem.discard(x)
                        if len(in_mem) < b:
                            break
            ops.append(["load", v])
            in_mem.add(v)
        else:
            for p in ps:
                ensure_in_mem(p)
            ops.append(["compute", v])
            in_mem.add(v)
    
    # Process outputs
    for o in outputs:
        ensure_in_mem(o)
        if o in in_mem:
            ops.append(["store", o])
            stored.add(o)
            in_mem.discard(o)
        
        # Cleanup: delete vertices that are no longer needed for remaining outputs
        remaining_outputs = [x for x in outputs if x not in stored]
        if not remaining_outputs:
            break
        
        # Find all ancestors of remaining outputs
        needed = set()
        for ro in remaining_outputs:
            stack = [ro]
            while stack:
                u = stack.pop()
                if u in needed:
                    continue
                needed.add(u)
                for p in parents.get(u, []):
                    stack.append(p)
        
        # Delete vertices in in_mem that are not in needed
        for x in list(in_mem):
            if x not in needed:
                ops.append(["delete", x])
                in_mem.discard(x)
    
    return ops[:20000]

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 50  # Safety margin
    
    candidates = []
    
    # 1. Try layered optimal
    try:
        r = _pebble_layered_optimal(dag, fast_memory_budget)
        if r is not None:
            candidates.append(r)
    except Exception:
        pass
    
    # 2. Try greedy memory-aware
    try:
        r = _pebble_greedy_memory_aware(dag, fast_memory_budget)
        if r is not None:
            candidates.append(r)
    except Exception:
        pass
    
    # 3. Fallback to naive (might be invalid if budget is small)
    try:
        r = _pebble_naive(dag, fast_memory_budget)
        if r is not None:
            candidates.append(r)
    except Exception:
        pass
    
    if not candidates:
        return []
    
    # Validate and pick best
    valid_candidates = []
    for r in candidates:
        is_valid, transfers, _ = _simulate(r, dag, fast_memory_budget)
        if is_valid:
            valid_candidates.append((transfers, r))
    
    if valid_candidates:
        valid_candidates.sort(key=lambda x: x[0])
        return valid_candidates[0][1]
    
    # If none valid, return the one with fewest ops (heuristic)
    candidates.sort(key=lambda x: len(x))
    return candidates[0]
# EVOLVE-BLOCK-END
