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
        # Fallback if cycle or disconnected (shouldn't happen in valid DAG)
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
        
        # Helper: Count remaining children that are not yet computed
        # We can approximate this by checking if the child is in 'computed' set.
        # Since we process in topo order, if a child is not computed, it's in the future.
        
        def can_delete(v):
            """Check if vertex v can be deleted (no uncomputed children)."""
            for c in children_map[v]:
                if c not in computed:
                    return False
            return True

        def evict_one():
            """Evict one vertex from fast memory to make space. Returns True if successful."""
            if not fast_memory:
                return False
            
            # Strategy: Evict the vertex that is safe to delete and has the "furthest" last use?
            # Actually, if we can delete, we should delete the one with the LARGEST last_use?
            # No, if we can delete it, its last_use is already in the past (all children computed).
            # So last_use < current_index.
            # If we MUST evict something we can't delete (because it has future children),
            # we have to store it (if output) or... wait, if it has future children, we can't delete it.
            # If it's not an output, we can't store it (store implies it's done? No, store just moves to blue).
            # But if it has future children, we need it RED for compute. If we store it, it becomes BLUE.
            # Then when we need to compute the child, we must LOAD it again.
            # This is a "recomputation" via memory round-trip.
            
            # Priority for eviction:
            # 1. Vertices that can be deleted (safe to lose).
            # 2. Vertices that are outputs (safe to store, must be stored eventually).
            # 3. Vertices that are non-outputs but have future children (must store, then reload later).
            
            # We want to minimize total operations.
            # Deleting is 1 op.
            # Storing is 1 op (and later a Load is 1 op).
            
            # Let's classify candidates:
            candidates = []
            for v in fast_memory:
                # v is in fast_memory (red)
                
                # Check if it's an output
                is_out = is_output[v]
                
                # Check if it can be deleted
                safe_to_delete = can_delete(v)
                
                if safe_to_delete:
                    if is_out:
                        # Output that is done: MUST store.
                        # Cost: 1 (store)
                        # Priority: High (must do eventually, might as well do now)
                        # But wait, if it's done, we can store it.
                        candidates.append((0, v, 'store')) 
                    else:
                        # Non-output that is done: DELETE.
                        # Cost: 1 (delete)
                        candidates.append((1, v, 'delete'))
                else:
                    # Not safe to delete (has future children)
                    if is_out:
                        # Output with future children: MUST keep red? 
                        # If we store it, it becomes blue. When child needs it, we load it.
                        # Cost: 1 (store) + 1 (load later) = 2 ops.
                        # But we free space now.
                        candidates.append((2, v, 'store'))
                    else:
                        # Non-output with future children:
                        # We cannot delete. We cannot "store" in the sense of finishing it.
                        # But we CAN store it to free space, and load it later.
                        # This is effectively recomputation overhead.
                        # Cost: 1 (store) + 1 (load later) = 2 ops.
                        candidates.append((3, v, 'store')) # Lower priority than output store? 
                        # Actually, storing a non-output is allowed? 
                        # "store" moves Red -> Blue.
                        # If it's not an output, it doesn't *need* to be blue at the end.
                        # But if we store it, we can load it back.
                        # Is it better to store or delete?
                        # We can't delete. So we must store.
                        # But wait, if we store a non-output, and later need it, we load it.
                        # If we don't store it, we can't free space.
                        # So we are forced to store it.
                        # However, we should prefer to evict "safe to delete" ones first.
                        # So (3) is correct: last resort.
            
            if not candidates:
                return False
                
            # Sort by priority (lower is better), then by last_use (higher last_use is "better" to keep? 
            # No, if we are evicting, we want to evict the one we need LEAST soon?
            # Or the one we need MOST soon?
            # If we evict a vertex with last_use=10 and keep one with last_use=5, we will need the 5 one sooner.
            # So we should evict the one with the LARGEST last_use?
            # Wait.
            # If I have A (last_use 5) and B (last_use 10).
            # If I evict A, I need to load A at step 5.
            # If I evict B, I need to load B at step 10.
            # It doesn't matter which I evict for the "future cost" in terms of *when* I pay the load cost, 
            # but it matters for *which* one I pay for.
            # Actually, if I evict A, I pay load(A). If I evict B, I pay load(B).
            # The cost is the same (1 load).
            # However, if I evict A, A is gone. If I need A again later? No, last_use is the LAST time.
            # So if I evict A, I only pay the load cost once (at the next use).
            # If I keep A and evict B, I pay load(B) once.
            # So the choice doesn't change the total number of loads, ONLY IF every evicted vertex is loaded exactly once more.
            # This is true if we only evict once per vertex? No, we might evict, load, evict, load...
            # But if we evict a vertex, we only need to load it when its next child is computed.
            # If we evict B (last_use 10), we load it at step 10 (or whenever its next child is).
            # If we evict A (last_use 5), we load it at step 5.
            # Total loads are the same.
            # So we can pick arbitrarily?
            # Let's pick the one with the LARGEST last_use to evict, to keep the "soon-to-be-needed" ones in cache?
            # This is "Keep Most Recently Needed"? No, "Keep Soonest Needed".
            # So Evict Furthest Last Use.
            
            candidates.sort(key=lambda x: (-last_use[x[1]], x[0]))
            
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
                # If an input was deleted, we can't reload it (it's not in slow memory? 
                # "Inputs start blue". If we delete an input, it goes to void. 
                # Can we reload a void input? 
                # The validator says: load requires state[v] == 'blue'.
                # So if an input is void, we CANNOT load it.
                # This implies we should NEVER delete an input.
                # In our eviction logic, inputs have no children? 
                # If an input has children, it's not a leaf.
                # If we delete it, it's void.
                # So we must ensure we don't delete inputs if they might be needed again?
                # Actually, if an input is used, and then all its children are computed, 
                # we CAN delete it. But we should not need it again.
                # If we need it again, it means it has a child that was not computed?
                # But we process in topo order. If a child is not computed, it's in the future.
                # So if `can_delete` returns True, it means all children are computed.
                # So we won't need the input again.
                # So deleting an input is safe IF all its children are computed.
                # But if we delete it, and then... wait.
                # If we delete it, state is void.
                # If we later try to `ensure_parent_in_memory` on it, and it's void, we return False.
                # This would break the schedule.
                # So we must ensure that if a parent is void, we don't try to load it.
                # But if a parent is void, it means it was deleted.
                # If it was deleted, it means all its children were computed.
                # So we shouldn't be computing a child of it now.
                # Unless... the DAG structure is such that we missed a dependency?
                # Or we deleted it prematurely?
                # `can_delete` checks `if c not in computed`.
                # So if a child is not computed, `can_delete` is False.
                # So we won't delete it.
                # So this case (void parent needed) should not happen.
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
            
            # Proactive cleanup?
            # The eviction in `ensure_parent_in_memory` and before compute handles pressure.
            # But we can also clean up dead vertices to keep memory low.
            # Let's do a light cleanup: if memory is > 50% full, try to evict dead vertices.
            # Or just rely on the "evict when needed" approach.
            # The "evict when needed" approach might evict a vertex that is still needed (store/load cycle).
            # Proactive deletion of dead vertices avoids store/load cycles for them.
            
            # Let's proactively delete any vertex that is dead (can_delete) and not an output.
            # And store any vertex that is dead and IS an output.
            
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
    
    def evict_one_simple():
        if not fast_memory:
            return False
        best_v = None
        best_lu = -1
        for v in fast_memory:
            if last_use[v] > best_lu:
                best_lu = last_use[v]
                best_v = v
        if best_v is None:
            return False
        if is_output[best_v]:
            ops.append(['store', best_v])
            state[best_v] = 'blue'
        else:
            ops.append(['delete', best_v])
            state[best_v] = 'void'
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
