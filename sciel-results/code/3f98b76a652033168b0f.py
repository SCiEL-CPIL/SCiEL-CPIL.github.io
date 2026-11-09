# EVOLVE-BLOCK-START
import time
import sys
import heapq
from collections import defaultdict

def _parse_dag(dag):
    """Parse various possible DAG formats into a unified structure."""
    # Try to extract: vertices, parents dict, outputs, inputs
    if isinstance(dag, dict):
        # Common keys: 'parents', 'outputs', 'inputs', 'vertices', 'n'
        parents = dag.get('parents', None)
        outputs = dag.get('outputs', None)
        inputs = dag.get('inputs', None)
        vertices = dag.get('vertices', None)
        n = dag.get('n', None)
        
        if parents is None:
            # Maybe it's a list of parent lists
            if isinstance(dag, list):
                parents = [list(p) for p in dag]
                n = len(dag)
                if outputs is None:
                    outputs = [n - 1]  # last vertex
                if inputs is None:
                    inputs = [i for i in range(n) if not parents[i]]
            else:
                return None, None, None, None, None
        else:
            if isinstance(parents, list):
                parents = [list(p) for p in parents]
                n = len(parents)
            elif isinstance(parents, dict):
                n = max(parents.keys()) + 1 if parents else 0
                parents_list = [list(parents.get(i, [])) for i in range(n)]
                parents = parents_list
            
            if outputs is None:
                # Outputs are vertices with no children
                children = [[] for _ in range(n)]
                for v in range(n):
                    for p in parents[v]:
                        children[p].append(v)
                outputs = [v for v in range(n) if not children[v]]
            
            if inputs is None:
                inputs = [v for v in range(n) if not parents[v]]
        
        return parents, outputs, inputs, vertices, n
    
    elif isinstance(dag, list):
        parents = [list(p) if isinstance(p, (list, tuple)) else [] for p in dag]
        n = len(dag)
        outputs = [n - 1] if n > 0 else []
        inputs = [i for i in range(n) if not parents[i]]
        return parents, outputs, inputs, None, n
    
    return None, None, None, None, None

def _greedy_pebble(parents, outputs, fast_memory_budget):
    """
    Greedy pebbling strategy for layered DAGs.
    Uses last_use heuristic to minimize transfers.
    """
    n = len(parents)
    if n == 0:
        return []
    
    # Precompute children
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
            
    # Precompute last_use: the last vertex index that depends on this vertex
    last_use = [v for v in range(n)]
    for v in range(n):
        if children[v]:
            last_use[v] = max(children[v])
            
    ops = []
    in_fast = set()
    is_output = set(outputs)
    
    # Track computed vertices. Inputs are considered "available" (computed=0 cost, but need load)
    # Actually, 'computed' in the state machine sense means the value exists.
    # Inputs exist in slow memory.
    
    for v in range(n):
        # 1. Load parents if not in fast memory
        for p in parents[v]:
            if p not in in_fast:
                ops.append(['load', p])
                in_fast.add(p)
                
        # 2. Compute v if it's not an input
        if parents[v]:
            ops.append(['compute', v])
            in_fast.add(v)
        else:
            # v is an input
            # If it's an output, we need to store it.
            # It might not be in fast memory yet (if it has no children or children not processed? 
            # Wait, if it has no children, it's an output? 
            # If it's an input and an output, we load and store.
            if v in is_output:
                if v not in in_fast:
                    ops.append(['load', v])
                    in_fast.add(v)
                ops.append(['store', v])
                in_fast.discard(v)
            # If it's not an output, we do nothing. It will be loaded when a child needs it.
            # But what if it has children? The first child will load it.
            # What if it has NO children and is NOT an output? Then it's irrelevant.
            
        # 3. Store v if it's an output (and was computed)
        if parents[v] and v in is_output:
            if v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)
                
        # 4. Eviction
        # Keep in_fast within budget.
        # Evict vertices that are no longer needed (last_use[u] <= v).
        # If still over budget, evict the one with the furthest last_use (max last_use).
        
        while len(in_fast) > fast_memory_budget:
            # Find dead vertices
            dead_vertices = [u for u in in_fast if last_use[u] <= v]
            if dead_vertices:
                # Delete any dead vertex
                u_to_delete = dead_vertices[0]
                ops.append(['delete', u_to_delete])
                in_fast.discard(u_to_delete)
            else:
                # All are live. Evict the one needed furthest in the future.
                # This minimizes the chance of needing to reload it soon.
                # Note: If two have same last_use, evict the one that is NOT an output? 
                # Outputs should have been stored. So all in_fast are non-outputs or not-yet-stored.
                # We store outputs immediately, so in_fast should not contain outputs.
                u_to_delete = max(in_fast, key=lambda u: last_use[u])
                ops.append(['delete', u_to_delete])
                in_fast.discard(u_to_delete)
                
    return ops

def _dijkstra_pebble(parents, outputs, fast_memory_budget, time_limit=80.0):
    """
    Exact state-space search using Dijkstra's algorithm.
    """
    n = len(parents)
    t0 = time.time()
    
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
            
    is_output = set(outputs)
    
    initial_computed = frozenset(v for v in range(n) if not parents[v])
    initial_state = (frozenset(), initial_computed, frozenset())
    goal_stored = frozenset(outputs)
    
    dist = {initial_state: 0}
    heap = [(0, initial_state)]
    parent = {}
    
    best_solution = None
    best_cost = float('inf')
    
    while heap:
        if time.time() - t0 > time_limit:
            break
        
        cost, state = heapq.heappop(heap)
        
        if cost > dist.get(state, float('inf')):
            continue
        
        in_fast, computed, stored = state
        
        if stored == goal_stored:
            best_cost = cost
            path = []
            curr = state
            while curr in parent:
                prev, op, v = parent[curr]
                path.append([op, v])
                curr = prev
            path.reverse()
            best_solution = path
            break
        
        # Successors
        # 1. Load
        for v in range(n):
            if v not in in_fast and v in computed and len(in_fast) < fast_memory_budget:
                new_in_fast = in_fast | {v}
                new_state = (new_in_fast, computed, stored)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'load', v)
                    heapq.heappush(heap, (new_cost, new_state))
        
        # 2. Compute
        for v in range(n):
            if v not in computed and v not in in_fast:
                all_parents_in_fast = all(p in in_fast for p in parents[v])
                if all_parents_in_fast and len(in_fast) < fast_memory_budget:
                    new_in_fast = in_fast | {v}
                    new_computed = computed | {v}
                    new_state = (new_in_fast, new_computed, stored)
                    new_cost = cost
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        parent[new_state] = (state, 'compute', v)
                        heapq.heappush(heap, (new_cost, new_state))
        
        # 3. Store
        for v in in_fast:
            if v in is_output and v not in stored:
                new_in_fast = in_fast - {v}
                new_stored = stored | {v}
                new_state = (new_in_fast, computed, new_stored)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'store', v)
                    heapq.heappush(heap, (new_cost, new_state))
        
        # 4. Delete
        for v in in_fast:
            if v not in is_output or v in stored:
                new_in_fast = in_fast - {v}
                new_state = (new_in_fast, computed, stored)
                new_cost = cost
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'delete', v)
                    heapq.heappush(heap, (new_cost, new_state))
    
    if best_solution is not None:
        return best_solution
    return None

def _validate_ops(ops, parents, outputs, fast_memory_budget):
    """Validate that operations are legal and complete."""
    n = len(parents)
    in_fast = set()
    computed = set()
    stored = set()
    
    for op, v in ops:
        if op == 'load':
            if v in in_fast:
                continue
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            stored.add(v)
        elif op == 'compute':
            for p in parents[v]:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
            computed.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.discard(v)
        else:
            return False
    
    for o in outputs:
        if o not in stored:
            return False
    
    return True

def _count_transfers(ops):
    """Count the number of transfers (load + store operations)."""
    return sum(1 for op, v in ops if op in ('load', 'store'))

def pebble(dag, fast_memory_budget):
    import time
    t0 = time.time()
    deadline = 85.0
    
    # Parse the DAG
    parents, outputs, inputs, vertices, n = _parse_dag(dag)
    
    if parents is None or n is None:
        return []
    
    if outputs is None:
        outputs = [n - 1]
    
    if fast_memory_budget is None:
        fast_memory_budget = n
    
    if fast_memory_budget <= 0:
        fast_memory_budget = 1
    
    # Try Dijkstra for small DAGs (exact solution)
    elapsed = time.time() - t0
    if n <= 15 and fast_memory_budget <= 8:
        try:
            dijkstra_time_limit = max(1.0, deadline - elapsed - 5.0)
            ops = _dijkstra_pebble(parents, outputs, fast_memory_budget, time_limit=dijkstra_time_limit)
            if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
                return ops
        except Exception:
            pass
    
    # Try Greedy Pebbling (fast and good heuristic)
    try:
        ops = _greedy_pebble(parents, outputs, fast_memory_budget)
        if _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    
    # Fallback to naive (should not happen if Greedy is robust)
    try:
        ops = []
        in_fast = set()
        for v in range(n):
            for p in parents[v]:
                if p not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        for u in list(in_fast):
                            if u not in outputs:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                break
                    ops.append(['load', p])
                    in_fast.add(p)
            
            if parents[v]:
                ops.append(['compute', v])
                in_fast.add(v)
            elif v in outputs:
                if v not in in_fast:
                    ops.append(['load', v])
                    in_fast.add(v)
            
            if v in outputs:
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
                for p in parents[v]:
                    if p in in_fast and p not in outputs:
                        still_needed = False
                        for fv in range(v + 1, n):
                            if p in parents[fv]:
                                still_needed = True
                                break
                        if not still_needed:
                            ops.append(['delete', p])
                            in_fast.discard(p)
        
        if _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    
    return []

# EVOLVE-BLOCK-END
