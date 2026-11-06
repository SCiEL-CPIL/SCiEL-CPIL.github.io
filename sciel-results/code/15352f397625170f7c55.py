"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque, defaultdict

def _topological_sort(vertices, parents):
    """
    Returns a list of vertices in topological order.
    vertices: list of vertex IDs
    parents: dict mapping vertex -> list of parent IDs
    """
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
        return vertices
    return order

def _validate_schedule(dag, ops, fast_memory_budget):
    """
    Validates a schedule by simulating the state machine.
    Returns True if valid, False otherwise.
    """
    vertices = set(dag['vertices'])
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    inputs = set(v for v in vertices if not parents.get(v, []))
    
    blue = set(inputs)
    red = set()
    computed = set()
    
    for op, v in ops:
        if v not in vertices:
            return False
        
        if op == 'load':
            if v not in blue or v in red:
                return False
            if len(red) >= fast_memory_budget:
                return False
            red.add(v)
            blue.remove(v)
            
        elif op == 'store':
            if v not in red or v in blue:
                return False
            red.remove(v)
            blue.add(v)
            
        elif op == 'compute':
            if v in red:
                pass # Recompute allowed
            
            for p in parents.get(v, []):
                if p not in red:
                    return False
            
            if v not in red:
                if len(red) >= fast_memory_budget:
                    return False
                red.add(v)
            computed.add(v)
            
        elif op == 'delete':
            if v not in red:
                return False
            red.remove(v)
            
        else:
            return False
            
    for v in outputs:
        if v not in blue:
            return False
            
    return True

def _robust_schedule(dag, fast_memory_budget):
    """
    Robust schedule:
    1. Topological order.
    2. Load inputs on demand (when first child is computed).
    3. Compute vertices.
    4. Immediately store/delete parents when their last child is computed.
    5. If memory is full before computing, evict any vertex with 0 remaining children.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    # Precompute children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # remaining_children[v] = number of children of v that have NOT been computed yet
    remaining_children = {v: len(children[v]) for v in vertices}
    
    ops = []
    in_memory = set()
    
    # Identify inputs
    inputs = set(v for v in vertices if not parents.get(v, []))
    
    for v in order:
        if v in inputs:
            continue # Inputs are handled when their children are computed
        
        # 1. Ensure parents are in memory
        for p in parents[v]:
            if p not in in_memory:
                if p in inputs:
                    # Load input on demand
                    if len(in_memory) >= fast_memory_budget:
                        # Need to free space before loading
                        freed = False
                        for u in list(in_memory):
                            if remaining_children[u] == 0:
                                if u in outputs:
                                    ops.append(['store', u])
                                else:
                                    ops.append(['delete', u])
                                in_memory.remove(u)
                                freed = True
                                break
                        if not freed:
                            return None # Cannot fit even one more
                    ops.append(['load', p])
                    in_memory.add(p)
                else:
                    # Parent should be in memory if it's not an input
                    # If it's not, something went wrong (e.g., it was deleted prematurely)
                    # This shouldn't happen if we only delete when remaining_children == 0
                    # and a parent of a current vertex always has remaining_children >= 1
                    return None
        
        # 2. Check if we have space for v
        if len(in_memory) >= fast_memory_budget:
            # Find a victim to evict
            # A victim is a vertex in memory with 0 remaining children
            # Note: v is not in memory yet
            freed = False
            for u in list(in_memory):
                if remaining_children[u] == 0:
                    if u in outputs:
                        ops.append(['store', u])
                    else:
                        ops.append(['delete', u])
                    in_memory.remove(u)
                    freed = True
                    break
            
            if not freed:
                # Cannot free space, schedule fails
                return None

        # 3. Compute v
        ops.append(['compute', v])
        in_memory.add(v)
        
        # 4. Update remaining children for parents and evict them if done
        for p in parents[v]:
            remaining_children[p] -= 1
            if remaining_children[p] == 0:
                # p is no longer needed
                if p in in_memory:
                    if p in outputs:
                        ops.append(['store', p])
                    else:
                        ops.append(['delete', p])
                    in_memory.remove(p)
    
    # 5. Final cleanup
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
            
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    try:
        # Try the robust schedule
        schedule = _robust_schedule(dag, fast_memory_budget)
        
        if schedule is not None:
            # Validate before returning
            if _validate_schedule(dag, schedule, fast_memory_budget):
                return schedule
            else:
                # Validation failed, try fallback
                pass
        
        # Fallback to baseline if robust schedule failed or was invalid
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
            
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

# EVOLVE-BLOCK-END
