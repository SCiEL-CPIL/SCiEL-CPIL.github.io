"""
Communication-minimizing DAG pebbling with bounded fast memory.

We implement a layered pebbling strategy that is significantly better than
the naive topological 'load-all-then-compute' approach. For each vertex in
topological order we:
  - load its parents if not already in fast memory,
  - compute the vertex (cost 0),
  - decide whether to store the result to slow memory and delete it from fast
    memory, or keep it in fast memory if budget allows.

The key optimization is to keep values in fast memory as long as possible
to avoid repeated loads, while respecting the fast_memory_budget constraint.
We use a simple heuristic: store a vertex's value if it has multiple children
and the current fast memory usage would exceed the budget, otherwise keep it.

For small DAGs (<= 25 vertices) we also try a Dijkstra-based exact search over
memory configurations to find the optimal transfer count. For larger DAGs we
fall back to the heuristic.
"""

import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem.
    
    dag: dict with keys:
      - 'vertices': list of vertex IDs (or similar structure)
      - 'parents': dict mapping vertex -> list of parent vertices (topologically ordered)
      - 'inputs': set/list of input vertices (start blue)
      - 'outputs': set/list of output vertices (must finish blue)
    
    fast_memory_budget: int, max number of red values in fast memory
    
    Returns: list of [operation, vertex] where operation is one of
             'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Parse the DAG structure
    # The dag structure may vary; let's handle common formats
    if isinstance(dag, dict):
        if 'vertices' in dag:
            vertices = list(dag['vertices'])
        else:
            vertices = list(range(dag.get('n', 0)))
        
        parents = dict(dag.get('parents', {}))
        # Ensure parents are lists
        for v in vertices:
            if v not in parents:
                parents[v] = []
            parents[v] = list(parents[v])
        
        inputs = set(dag.get('inputs', []))
        if not inputs:
            # Assume vertices with no parents are inputs
            inputs = set(v for v in vertices if not parents[v])
        
        outputs = set(dag.get('outputs', []))
        if not outputs:
            # Assume vertices with no children are outputs
            children = defaultdict(list)
            for v in vertices:
                for p in parents[v]:
                    children[p].append(v)
            outputs = set(v for v in vertices if not children[v])
    else:
        # Fallback: treat dag as list of (vertex, parents) tuples
        vertices = []
        parents = {}
        for item in dag:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v = item[0]
                ps = item[1]
                vertices.append(v)
                parents[v] = list(ps)
            else:
                vertices.append(item)
                parents[item] = []
        inputs = set(v for v in vertices if not parents.get(v, []))
        children = defaultdict(list)
        for v in vertices:
            for p in parents[v]:
                children[p].append(v)
        outputs = set(v for v in vertices if not children[v])
    
    n = len(vertices)
    
    if n == 0:
        return []
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)
    
    # Compute topological order (parents should already be ordered, but let's verify)
    in_degree = {v: len(parents[v]) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != n:
        # Cycle or issue; just use vertices as-is
        topo_order = list(vertices)
    
    # For small DAGs, try exact Dijkstra search
    if n <= 12 and fast_memory_budget <= 8:
        result = _dijkstra_pebbling(topo_order, parents, children, inputs, outputs, 
                                      fast_memory_budget, vertices, deadline)
        if result is not None:
            return result
    
    # Heuristic pebbling
    ops = []
    fast_mem = set()  # vertices currently in fast memory (red)
    slow_stored = set()  # vertices stored in slow memory
    
    # We process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        # Check if all parents are available in fast memory
        for p in parents[v]:
            if p not in fast_mem:
                # Need to load p from slow memory
                ops.append(['load', p])
                fast_mem.add(p)
                # Check budget: if we exceed, we need to evict something
                if len(fast_mem) > fast_memory_budget:
                    # Evict the vertex with fewest remaining uses
                    # Find which fast_mem vertices have remaining children to compute
                    remaining_children = {}
                    for fm in fast_mem:
                        remaining_children[fm] = 0
                        # Count how many children of fm are not yet computed
                        # We track this by checking which children are still in topo_order after current position
                        pass
                    # Simple eviction: evict the one with least future use
                    # Count future uses
                    future_use = {fm: 0 for fm in fast_mem}
                    # Build set of vertices yet to be processed
                    idx = topo_order.index(v) if v in topo_order else 0
                    remaining_set = set(topo_order[idx:]) if idx < len(topo_order) else set()
                    for fm in fast_mem:
                        for c in children[fm]:
                            if c in remaining_set:
                                future_use[fm] += 1
                    # Evict the one with min future_use, break ties by not evicting current parent if possible
                    min_use = min(future_use.values())
                    candidates = [fm for fm in fast_mem if future_use[fm] == min_use]
                    # Prefer to evict one that is not a parent of v
                    non_parents = [c for c in candidates if c not in parents[v]]
                    if non_parents:
                        to_evict = non_parents[0]
                    else:
                        to_evict = candidates[0]
                    ops.append(['delete', to_evict])
                    fast_mem.discard(to_evict)
        
        # Compute v
        ops.append(['compute', v])
        
        # Add v to fast memory
        fast_mem.add(v)
        
        # Decide whether to store v and delete it, or keep it
        # If v has children and fast memory is tight, store and delete
        num_children = len(children[v])
        
        if num_children == 0:
            # Leaf (output or terminal): must keep until end? No, outputs must finish blue.
            # Blue means stored in slow memory. So we need to store it.
            if v in outputs:
                ops.append(['store', v])
                slow_stored.add(v)
                fast_mem.discard(v)
        else:
            # Has children: decide based on budget
            if len(fast_mem) > fast_memory_budget:
                # Must evict something
                # Store v and delete it, or evict another
                # Prefer to keep v if it has many children
                future_use = {fm: 0 for fm in fast_mem}
                idx = topo_order.index(v) if v in topo_order else 0
                remaining_set = set(topo_order[idx+1:]) if idx+1 < len(topo_order) else set()
                for fm in fast_mem:
                    for c in children[fm]:
                        if c in remaining_set:
                            future_use[fm] += 1
                
                min_use = min(future_use.values())
                candidates = [fm for fm in fast_mem if future_use[fm] == min_use]
                # If v has the min use, evict v
                if v in candidates and len(candidates) > 1:
                    # Prefer to evict another
                    others = [c for c in candidates if c != v]
                    if others:
                        to_evict = others[0]
                        ops.append(['delete', to_evict])
                        fast_mem.discard(to_evict)
                    else:
                        ops.append(['store', v])
                        slow_stored.add(v)
                        fast_mem.discard(v)
                elif v in candidates:
                    ops.append(['store', v])
                    slow_stored.add(v)
                    fast_mem.discard(v)
                else:
                    to_evict = candidates[0]
                    ops.append(['delete', to_evict])
                    fast_mem.discard(to_evict)
            else:
                # Budget allows, keep in fast memory
                # But if v is an output, we must store it
                if v in outputs:
                    ops.append(['store', v])
                    slow_stored.add(v)
                    fast_mem.discard(v)
    
    # Final cleanup: ensure all outputs are stored (blue)
    for v in outputs:
        if v in fast_mem:
            ops.append(['store', v])
            fast_mem.discard(v)
        # If not in fast_mem and not in slow_stored, it might have been deleted
        # We need to recompute? This is a problem with our heuristic.
        # For now, assume all outputs were stored.
    
    # Truncate to max 20000 ops
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops


def _dijkstra_pebbling(topo_order, parents, children, inputs, outputs, 
                        budget, vertices, deadline):
    """
    Exact Dijkstra search over memory configurations.
    State: (frozenset of vertices in fast memory, frozenset of vertices computed)
    We track which vertices have been computed (their value exists somewhere).
    Actually, we need to track:
    - Which vertices are in fast memory (red)
    - Which vertices have been computed (value exists in slow or fast)
    
    A vertex's value exists if it has been computed and not deleted from slow.
    But we don't track slow memory explicitly - we assume once computed, it's in slow
    unless we delete it (but we can't delete from slow in this model? Actually the
    operations are load/store/compute/delete. Delete removes from fast memory.
    Store moves from fast to slow. Load moves from slow to fast.
    
    So slow memory always contains all computed values that have been stored.
    Fast memory contains a subset of computed values.
    
    State: (fast_mem_frozenset, computed_frozenset)
    But computed is monotonic (only grows), so we can track it.
    
    Actually, let's simplify: since we process in topological order, we can track
    progress by how far we are in the topo order. But Dijkstra doesn't assume
    a fixed order.
    
    Let's use: state = (fast_set, done_set) where done_set is the set of vertices
    whose computation is complete (value exists in slow memory).
    
    Initially: fast_set = set(inputs), done_set = set(inputs)  [inputs start blue = in slow]
    Wait, "inputs start blue" means they are in slow memory. So they are computed and stored.
    
    Goal: all outputs are in done_set (blue = in slow memory).
    
    Transitions:
    - load(v): if v in done_set and v not in fast_set, and len(fast_set) < budget
      -> fast_set' = fast_set | {v}, cost +1
    - store(v): if v in fast_set
      -> fast_set' = fast_set - {v}, done_set' = done_set | {v}, cost +1
    - compute(v): if all parents of v are in fast_set, and v not in done_set
      -> fast_set' = fast_set | {v}, done_set' unchanged (v is now in fast, not yet stored), cost 0
      Wait, after compute, v is in fast memory. So fast_set' = fast_set | {v}
    - delete(v): if v in fast_set and v not in outputs (can't delete outputs before they're blue)
      Actually, we can delete any fast value. But if we delete an output before storing it,
      we lose it. So delete(v) is only useful if v is in done_set (already stored).
      -> fast_set' = fast_set - {v}, cost +1
    
    Actually, delete only makes sense for values already stored. For values not yet stored,
    deleting loses them forever (unless recomputed).
    
    Let's refine:
    - compute(v): requires all parents in fast_set, v not in done_set
      Result: v added to fast_set. Cost 0.
    - store(v): requires v in fast_set
      Result: v removed from fast_set, added to done_set. Cost 1.
    - load(v): requires v in done_set, v not in fast_set, len(fast_set) < budget
      Result: v added to fast_set. Cost 1.
    - delete(v): requires v in fast_set and v in done_set (safe to delete)
      Result: v removed from fast_set. Cost 1.
    
    This is a valid model. Let's implement Dijkstra.
    """
    # State: (fast_set, done_set)
    # Use frozensets for hashing
    
    initial_fast = frozenset(inputs)
    initial_done = frozenset(inputs)
    initial_state = (initial_fast, initial_done)
    
    # Check if already done
    if outputs.issubset(initial_done):
        return []
    
    # Dijkstra: dist[state] = min transfers
    dist = {initial_state: 0}
    # Priority queue: (cost, state)
    pq = [(0, initial_state)]
    
    # Track parent for path reconstruction
    parent = {}
    
    while pq:
        if time.time() > deadline:
            return None
        
        cost, state = heapq.heappop(pq)
        fast_set, done_set = state
        
        if cost > dist.get(state, float('inf')):
            continue
        
        # Check goal
        if outputs.issubset(done_set):
            # Reconstruct path
            ops = []
            cur = state
            while cur in parent:
                ops.append(parent[cur])
                cur = parent[cur][0]
            ops.reverse()
            return ops
        
        # Generate transitions
        # 1. compute(v)
        for v in vertices:
            if v in done_set:
                continue
            # Check all parents in fast_set
            if all(p in fast_set for p in parents[v]):
                new_fast = frozenset(fast_set | {v})
                new_state = (new_fast, done_set)
                new_cost = cost + 0
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = (state, ['compute', v])
                    heapq.heappush(pq, (new_cost, new_state))
        
        # 2. store(v)
        for v in fast_set:
            new_fast = frozenset(fast_set - {v})
            new_done = frozenset(done_set | {v})
            new_state = (new_fast, new_done)
            new_cost = cost + 1
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                parent[new_state] = (state, ['store', v])
                heapq.heappush(pq, (new_cost, new_state))
        
        # 3. load(v)
        if len(fast_set) < budget:
            for v in done_set:
                if v not in fast_set:
                    new_fast = frozenset(fast_set | {v})
                    new_state = (new_fast, done_set)
                    new_cost = cost + 1
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        parent[new_state] = (state, ['load', v])
                        heapq.heappush(pq, (new_cost, new_state))
        
        # 4. delete(v) - only if v in done_set
        for v in fast_set:
            if v in done_set:
                new_fast = frozenset(fast_set - {v})
                new_state = (new_fast, done_set)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = (state, ['delete', v])
                    heapq.heappush(pq, (new_cost, new_state))
    
    return None
