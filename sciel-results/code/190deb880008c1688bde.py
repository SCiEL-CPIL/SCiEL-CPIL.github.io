"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque, defaultdict

def _topological_sort(vertices, parents):
    """Returns a list of vertices in topological order."""
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in in_degree:
                    in_degree[v] += 1
                    children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(order) != len(vertices):
        return list(vertices)
    return order

def _analyze_dag(dag):
    """
    Pre-compute DAG properties:
    - Topological order
    - Children map
    - Last use index for each vertex (index in topo order of its last child)
    - Is input / Is output
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    order_index = {v: i for i, v in enumerate(order)}
    
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children_map:
                children_map[p].append(v)
    
    is_input = {v: (len(parents.get(v, [])) == 0) for v in vertices}
    is_output = {v: (v in outputs) for v in vertices}
    
    # Calculate last use index
    last_use = {}
    for v in vertices:
        if children_map[v]:
            last_use[v] = max(order_index[c] for c in children_map[v])
        else:
            last_use[v] = order_index[v]
            
    return order, order_index, children_map, is_input, is_output, last_use

def _validate_schedule(dag, ops, fast_memory_budget):
    """Validate that the schedule is correct and respects memory budget."""
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    state = {}
    for v in vertices:
        if not parents.get(v, []):
            state[v] = 'blue'
        else:
            state[v] = 'void'
    
    fast_memory = set()
    computed = set()
    
    for op, v in ops:
        if op == 'load':
            if state[v] != 'blue':
                return False
            if v in fast_memory:
                return False
            if len(fast_memory) >= fast_memory_budget:
                return False
            fast_memory.add(v)
            state[v] = 'red'
        elif op == 'store':
            if state[v] != 'red':
                return False
            if v not in fast_memory:
                return False
            fast_memory.discard(v)
            state[v] = 'blue'
        elif op == 'delete':
            if state[v] != 'red':
                return False
            if v not in fast_memory:
                return False
            fast_memory.discard(v)
            state[v] = 'void'
        elif op == 'compute':
            if v in computed:
                return False
            for p in parents.get(v, []):
                if state.get(p) != 'red':
                    return False
            if len(fast_memory) >= fast_memory_budget:
                return False
            fast_memory.add(v)
            state[v] = 'red'
            computed.add(v)
        else:
            return False
    
    # Check all outputs are blue
    for v in outputs:
        if state.get(v) != 'blue':
            return False
            
    return True

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    try:
        vertices = dag['vertices']
        parents = dag['parents']
        outputs = set(dag['outputs'])
        
        if not vertices:
            return []
            
        order, order_index, children_map, is_input, is_output, last_use = _analyze_dag(dag)
        
        # State tracking
        state = {}
        for v in vertices:
            if is_input[v]:
                state[v] = 'blue'
            else:
                state[v] = 'void'
                
        fast_memory = set()
        ops = []
        computed = set()
        
        # Pre-compute "remaining children" counts for O(1) can_delete check
        # remaining_children[v] = number of children of v that are not yet computed
        remaining_children = {v: len(children_map[v]) for v in vertices}
        
        def can_delete(v):
            """Check if vertex v can be deleted (no uncomputed children)."""
            return remaining_children[v] == 0

        def evict_one():
            """Evict one vertex from fast memory to make space. Returns True if successful."""
            if not fast_memory:
                return False
            
            # Priority for eviction:
            # 1. Vertices that can be deleted (safe to lose).
            #    - If output: Store (must be blue at end). Cost 1.
            #    - If not output: Delete. Cost 1.
            # 2. Vertices that are outputs (safe to store, must be stored eventually).
            #    - Store. Cost 1 now, 1 load later.
            # 3. Vertices that are non-outputs but have future children.
            #    - Store. Cost 1 now, 1 load later.
            
            # We want to minimize total operations.
            # Deleting is 1 op.
            # Storing is 1 op (and later a Load is 1 op).
            
            # Let's classify candidates:
            # We prefer to evict vertices that are "dead" (can_delete) because we don't pay the reload cost.
            # Among dead vertices, prefer non-outputs (delete) or outputs (store)?
            # Both cost 1 op now. Deleting a non-output is final. Storing an output is final (for that value).
            # So dead vertices are best.
            
            # Among live vertices (cannot delete), we must store them.
            # Which one to store?
            # We want to minimize future loads.
            # If we store v, we will load it when its next child is computed.
            # So the "cost" is 1 load later.
            # Is there a difference between storing v1 vs v2?
            # Only if one of them is an output? No, outputs must be stored anyway.
            # So for live vertices, it doesn't matter much, but we might prefer to keep the one with the SOONEST next use?
            # Or the one with the LARGEST remaining use?
            
            # Let's refine the heuristic:
            # 1. Dead non-outputs: Delete. (Best)
            # 2. Dead outputs: Store. (Good)
            # 3. Live non-outputs: Store. (Bad, reload later)
            # 4. Live outputs: Store. (Bad, reload later, but must store eventually)
            
            # Within each category, which one?
            # For live vertices, if we store v, we pay 1 load later.
            # If we store v1 (next use in 5 steps) and v2 (next use in 10 steps).
            # It doesn't change the total number of loads.
            # However, if we store v1, v1 is out of cache. If we need v1 again? No, last_use is the last.
            # So any live vertex we evict will be loaded exactly once more (at its next use).
            # So the choice among live vertices doesn't affect the total count of loads.
            # But it might affect memory pressure in the future?
            # If we keep v1 (soon use) and evict v2 (far use), we avoid loading v1 soon, but load v2 later.
            # If we evict v1 (soon use) and keep v2 (far use), we load v1 soon, but keep v2.
            # The total loads are the same.
            # However, keeping the "soon use" vertex might be better for locality?
            # Actually, if we evict the "far use" vertex, we free space now, and we don't need that space until later.
            # So evicting the "far use" vertex seems safer?
            # Let's evict the vertex with the LARGEST last_use (furthest in the future) among live vertices.
            # This keeps the "soon-to-be-needed" vertices in cache.
            
            candidates = []
            
            # 1. Dead non-outputs
            for v in fast_memory:
                if can_delete(v) and not is_output[v]:
                    candidates.append((0, v, 'delete'))
            
            # 2. Dead outputs
            for v in fast_memory:
                if can_delete(v) and is_output[v]:
                    candidates.append((1, v, 'store'))
            
            # 3. Live non-outputs
            for v in fast_memory:
                if not can_delete(v) and not is_output[v]:
                    candidates.append((2, v, 'store'))
            
            # 4. Live outputs
            for v in fast_memory:
                if not can_delete(v) and is_output[v]:
                    candidates.append((3, v, 'store'))
            
            if not candidates:
                return False
                
            # Sort by priority, then by last_use (descending for live, doesn't matter for dead?)
            # For dead, last_use is in the past.
            # For live, we want to evict the one with the LARGEST last_use (furthest future).
            # So sort by priority ASC, then last_use DESC.
            
            candidates.sort(key=lambda x: (x[0], -last_use[x[1]]))
            
            _, v, action = candidates[0]
            
            if action == 'delete':
                ops.append(['delete', v])
                state[v] = 'void'
            else: # store
                ops.append(['store', v])
                state[v] = 'blue'
            
            fast_memory.discard(v)
            return True

        def ensure_parent_in_memory(p):
            """Ensure parent p is in fast memory (red)."""
            if state[p] == 'red':
                return True
            
            if state[p] == 'blue':
                # Need to load
                if len(fast_memory) >= fast_memory_budget:
                    # Try to evict
                    if not evict_one():
                        return False
                ops.append(['load', p])
                fast_memory.add(p)
                state[p] = 'red'
                return True
            else:
                # state[p] == 'void'
                # This means p was deleted. We need to recompute it?
                # Or was it an input? Inputs start blue, so they can't be void unless deleted.
                # If an input was deleted, we can't reload it.
                # This case should not happen if we don't delete inputs prematurely.
                return False

        # Main Loop
        for v in order:
            # Check time
            if time.time() > deadline:
                break
            
            if is_input[v]:
                # Inputs start blue. We don't compute them.
                # Do we need to load them?
                # Only when a child needs them.
                # So we don't load them here.
                computed.add(v)
                # Update remaining children counts for parents of v?
                # No, v is an input, it has no parents.
                # But v is a parent of its children.
                # We don't update remaining_children here because v is not being "computed" in the sense of producing a value for the first time?
                # Actually, inputs are "computed" at start.
                # So we should update remaining_children for inputs?
                # No, remaining_children tracks how many children of a vertex are uncomputed.
                # When we "compute" a vertex v, we decrement remaining_children for all parents of v.
                # For inputs, we don't have parents.
                # But we do mark them as computed.
                # So if an input p has a child v, and v is computed, we decrement remaining_children[p].
                # This happens in the main loop when v is processed.
                continue
            
            # Non-input vertex
            all_parents_ready = True
            for p in parents[v]:
                if state[p] != 'red':
                    if not ensure_parent_in_memory(p):
                        all_parents_ready = False
                        break
            
            if not all_parents_ready:
                # Failure state for this simple scheduler
                # This shouldn't happen if eviction logic is correct.
                # If it does, we might need a more complex recompute.
                # For now, we assume it works.
                pass

            # Compute v
            # Check memory space
            if len(fast_memory) >= fast_memory_budget:
                if not evict_one():
                    # Can't make space
                    pass

            ops.append(['compute', v])
            fast_memory.add(v)
            state[v] = 'red'
            computed.add(v)
            
            # Update remaining children counts for parents of v
            for p in parents[v]:
                remaining_children[p] -= 1
            
            # Proactive cleanup?
            # The eviction in `ensure_parent_in_memory` and before compute handles pressure.
            # But we can also clean up dead vertices to keep memory low.
            # Let's proactively delete any vertex that is dead (can_delete) and not an output.
            # And store any vertex that is dead and IS an output.
            
            # Only do this if memory is getting tight? Or always?
            # Doing it always keeps memory minimal, which is good for future evictions.
            # But it adds operations.
            # If we delete a dead vertex, we save a slot.
            # If we don't, we might have to evict it later (cost 1 op + 1 load if live, or 1 op if dead).
            # So proactive deletion of dead vertices is always good?
            # Yes, because it reduces memory pressure without future cost.
            
            for w in list(fast_memory):
                if time.time() > deadline:
                    break
                if can_delete(w):
                    if is_output[w]:
                        ops.append(['store', w])
                        state[w] = 'blue'
                    else:
                        ops.append(['delete', w])
                        state[w] = 'void'
                    fast_memory.discard(w)
            
        # Final Cleanup
        for v in list(fast_memory):
            if is_output[v]:
                ops.append(['store', v])
                fast_memory.discard(v)
                state[v] = 'blue'
            else:
                ops.append(['delete', v])
                fast_memory.discard(v)
                state[v] = 'void'
        
        # Validate the schedule
        if _validate_schedule(dag, ops, fast_memory_budget):
            if len(ops) <= 20000:
                return ops
        
        # Fallback
        return _safe_naive_schedule(dag, fast_memory_budget)

    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []

def _safe_naive_schedule(dag, fast_memory_budget):
    """
    A very conservative schedule.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    if not vertices:
        return []

    order, order_index, children_map, is_input, is_output, last_use = _analyze_dag(dag)
    
    state = {}
    for v in vertices:
        if is_input[v]:
            state[v] = 'blue'
        else:
            state[v] = 'void'
            
    fast_memory = set()
    ops = []
    computed = set()
    remaining_children = {v: len(children_map[v]) for v in vertices}
    
    def evict_one_simple():
        if not fast_memory:
            return False
        # Evict the vertex with the largest last_use (furthest in the future)
        # If it's dead, delete/store. If live, store.
        best_v = None
        best_lu = -1
        for v in fast_memory:
            if last_use[v] > best_lu:
                best_lu = last_use[v]
                best_v = v
        if best_v is None:
            return False
        if remaining_children[best_v] == 0:
            # Dead
            if is_output[best_v]:
                ops.append(['store', best_v])
                state[best_v] = 'blue'
            else:
                ops.append(['delete', best_v])
                state[best_v] = 'void'
        else:
            # Live
            ops.append(['store', best_v])
            state[best_v] = 'blue'
        fast_memory.discard(best_v)
        return True

    for v in order:
        if is_input[v]:
            computed.add(v)
            continue
            
        for p in parents[v]:
            if state[p] != 'red':
                if state[p] == 'blue':
                    while len(fast_memory) >= fast_memory_budget:
                        if not evict_one_simple():
                            break
                    ops.append(['load', p])
                    fast_memory.add(p)
                    state[p] = 'red'
        
        while len(fast_memory) >= fast_memory_budget:
            if not evict_one_simple():
                break
        
        ops.append(['compute', v])
        fast_memory.add(v)
        state[v] = 'red'
        computed.add(v)
        
        for p in parents[v]:
            remaining_children[p] -= 1
        
        if is_output[v] and not children_map[v]:
            ops.append(['store', v])
            fast_memory.discard(v)
            state[v] = 'blue'

    for v in list(fast_memory):
        if is_output[v]:
            ops.append(['store', v])
            fast_memory.discard(v)
            state[v] = 'blue'
        else:
            ops.append(['delete', v])
            fast_memory.discard(v)
            state[v] = 'void'
            
    return ops

# EVOLVE-BLOCK-END
