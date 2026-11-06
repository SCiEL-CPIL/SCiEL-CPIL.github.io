"""Communication-minimizing DAG pebbling for layered DAGs."""
import time

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Produce a pebbling schedule for a DAG.
    
    dag: dict with keys:
        - 'n': number of vertices
        - 'parents': list of lists, parents[v] = list of parent vertex indices
        - 'outputs': list of output vertex indices
        - possibly 'inputs': list of input vertex indices
    fast_memory_budget: int, max red pebbles allowed
    
    Returns: list of [operation, vertex] where operation in {'load','store','compute','delete'}
    """
    start_time = time.time()
    DEADLINE = 75.0  # seconds, safety margin under 90s kill
    
    n = dag['n']
    parents = dag.get('parents', [])
    outputs = dag.get('outputs', [])
    inputs = dag.get('inputs', [])
    
    # Build children map
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Determine which vertices are inputs (no parents)
    if not inputs:
        inputs = [v for v in range(n) if len(parents[v]) == 0]
    
    # Topological order (Kahn's algorithm)
    in_degree = [len(parents[v]) for v in range(n)]
    from collections import deque
    queue = deque([v for v in range(n) if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo) < n:
        # Cyclic or malformed; fall back to simple schedule
        topo = list(range(n))
    
    # Identify layers: a vertex's layer = longest path from any input
    layer = [0] * n
    for v in topo:
        if parents[v]:
            layer[v] = max(layer[p] for p in parents[v]) + 1
        # else layer[v] = 0 (input)
    
    max_layer = max(layer) if n > 0 else 0
    
    # Group vertices by layer
    layer_groups = [[] for _ in range(max_layer + 1)]
    for v in range(n):
        layer_groups[layer[v]].append(v)
    
    # Strategy: For each layer, we need all parents of that layer's vertices to be in fast memory.
    # We load parents, compute the layer, then delete parents that are no longer needed.
    # Key optimization: if a parent is also needed by the next layer, keep it.
    
    # Determine for each vertex, whether it's needed by any vertex in a later layer
    # (i.e., it has children in layer > current)
    
    # Simulate the schedule
    schedule = []
    in_fast = set()  # vertices currently in fast memory (red pebbles)
    
    # We process layer by layer
    # For each layer L, we need: all parents of vertices in layer L to be in fast memory
    # After computing layer L, we can delete parents that are NOT needed by layer L+1 or later
    
    # Precompute: for each vertex, the set of layers that need it as a parent
    needed_by_layer = [set() for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            needed_by_layer[p].add(layer[v])
    
    # For each layer, compute the set of parents needed
    for L in range(max_layer + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        verts_in_layer = layer_groups[L]
        if not verts_in_layer:
            continue
        
        # Collect all parents needed for this layer
        parents_needed = set()
        for v in verts_in_layer:
            for p in parents[v]:
                parents_needed.add(p)
        
        # Load any parents not already in fast memory
        for p in parents_needed:
            if p not in in_fast:
                schedule.append(['load', p])
                in_fast.add(p)
        
        # Compute all vertices in this layer
        for v in verts_in_layer:
            schedule.append(['compute', v])
            in_fast.add(v)
        
        # Now delete parents that are no longer needed
        # A parent p can be deleted if no vertex in layer > L needs it as a parent
        # AND p is not an output (outputs must finish blue, but we delete to free space)
        # Actually, outputs must be in fast memory at the end. So don't delete outputs.
        # Also, a vertex might be needed as a parent by a later layer.
        
        # Determine which parents can be deleted
        for p in list(parents_needed):
            # Can we delete p?
            # p is needed by layers in needed_by_layer[p]
            # If all layers needing p are <= L, we can delete it (after computing layer L)
            # But wait: p might be in a later layer itself (p is a vertex with its own layer)
            # p's layer is layer[p]. If layer[p] <= L, then p was computed in an earlier or same layer.
            # p is needed as parent by vertices in layers in needed_by_layer[p].
            # If max(needed_by_layer[p]) <= L, then p is no longer needed as a parent.
            # But p might still be an output, or needed for something else.
            
            # Actually, the key constraint: after this layer, is p needed as a parent by any future layer?
            future_need = any(l > L for l in needed_by_layer[p])
            
            # Can delete if: not in future need, not an output, and p is not in current layer (we just computed it)
            if not future_need and p not in outputs and p not in verts_in_layer:
                if p in in_fast:
                    schedule.append(['delete', p])
                    in_fast.discard(p)
    
    # At the end, all outputs must be in fast memory (blue). They are, since we never delete them.
    # But we should ensure all outputs are in_fast. They are, since we computed them and didn't delete.
    
    # Check budget: we might exceed fast_memory_budget at some point
    # For now, this is a heuristic. If budget is tight, we need more aggressive deletion.
    # Let's add a check and if we exceed, we do more deletions.
    
    # Verify and potentially fix: simulate to check max fast memory usage
    if time.time() - start_time > DEADLINE:
        return schedule
    
    # Simulate to check peak usage
    fast_count = 0
    peak = 0
    for op, v in schedule:
        if op == 'load':
            fast_count += 1
        elif op == 'compute':
            fast_count += 1
        elif op == 'delete':
            fast_count -= 1
        elif op == 'store':
            pass  # store doesn't change fast memory count? Actually store moves to slow, freeing fast
        peak = max(peak, fast_count)
    
    if peak > fast_memory_budget:
        # Need to be more aggressive with deletion
        # Re-do with stricter deletion: delete as soon as possible
        schedule = []
        in_fast = set()
        
        for L in range(max_layer + 1):
            if time.time() - start_time > DEADLINE:
                break
            
            verts_in_layer = layer_groups[L]
            if not verts_in_layer:
                continue
            
            parents_needed = set()
            for v in verts_in_layer:
                for p in parents[v]:
                    parents_needed.add(p)
            
            # Load parents
            for p in parents_needed:
                if p not in in_fast:
                    schedule.append(['load', p])
                    in_fast.add(p)
            
            # Compute
            for v in verts_in_layer:
                schedule.append(['compute', v])
                in_fast.add(v)
            
            # Delete parents aggressively: delete any parent that won't be needed again
            for p in sorted(parents_needed):
                future_need = any(l > L for l in needed_by_layer[p])
                if not future_need and p not in outputs and p not in verts_in_layer:
                    if p in in_fast:
                        schedule.append(['delete', p])
                        in_fast.discard(p)
        
        # Check again
        fast_count = 0
        peak = 0
        for op, v in schedule:
            if op in ('load', 'compute'):
                fast_count += 1
            elif op == 'delete':
                fast_count -= 1
            peak = max(peak, fast_count)
        
        if peak > fast_memory_budget:
            # Last resort: delete all non-output vertices after each layer
            schedule = []
            in_fast = set()
            for L in range(max_layer + 1):
                if time.time() - start_time > DEADLINE:
                    break
                verts_in_layer = layer_groups[L]
                if not verts_in_layer:
                    continue
                parents_needed = set()
                for v in verts_in_layer:
                    for p in parents[v]:
                        parents_needed.add(p)
                for p in parents_needed:
                    if p not in in_fast:
                        schedule.append(['load', p])
                        in_fast.add(p)
                for v in verts_in_layer:
                    schedule.append(['compute', v])
                    in_fast.add(v)
                # Delete all non-output, non-future-needed parents AND computed vertices from previous layers
                for p in list(in_fast):
                    if p not in outputs and p not in verts_in_layer:
                        # Check if p is needed as parent by any future layer
                        future_need = any(l > L for l in needed_by_layer[p])
                        if not future_need:
                            schedule.append(['delete', p])
                            in_fast.discard(p)
    
    # Final: ensure outputs are in fast memory (they should be)
    # Add any missing outputs via load if somehow not present (shouldn't happen)
    
    return schedule
# EVOLVE-BLOCK-END
