"""DAG pebbling solver with next-use-aware eviction and delete optimization."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', []))
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', dag.get('sinks', []))
        inputs = dag.get('inputs', dag.get('sources', []))
    elif isinstance(dag, (list, tuple)):
        vertices = list(range(len(dag)))
        parents = {}
        outputs = []
        inputs = []
        for i, v in enumerate(dag):
            if isinstance(v, dict):
                p = v.get('parents', v.get('deps', []))
                parents[i] = list(p) if p else []
            else:
                parents[i] = []
    else:
        return []
    
    if not vertices:
        return []
    
    n = len(vertices)
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological order
    in_degree = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            in_degree[v] += 1
    
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        return []
    
    topo_index = {v: i for i, v in enumerate(topo_order)}
    
    input_set = set(inputs) if inputs else set(v for v in vertices if len(parents.get(v, [])) == 0)
    output_set = set(outputs) if outputs else set(v for v in vertices if len(children[v]) == 0)
    
    # Children sorted by topo index for efficient next-use lookup
    children_by_topo = {v: sorted(children[v], key=lambda c: topo_index[c]) for v in vertices}
    
    # State
    state = {v: 'blue' if v in input_set else 'red' for v in vertices}
    fast_mem = set(v for v in vertices if state[v] == 'blue')
    operations = []
    
    # unprocessed_child_count[v] = number of children of v not yet processed
    unprocessed_child_count = {v: len(children[v]) for v in vertices}
    # processed_children[v] = set of children of v that have been processed
    processed_children = defaultdict(set)
    
    # For efficient next-use lookup, maintain a pointer for each vertex
    # next_child_ptr[v] = index into children_by_topo[v] of the first unprocessed child
    next_child_ptr = {v: 0 for v in vertices}
    
    def get_next_use(v):
        """Return topo_index of next unprocessed child of v, or None if no more."""
        ptr = next_child_ptr[v]
        child_list = children_by_topo[v]
        while ptr < len(child_list):
            c = child_list[ptr]
            if c not in processed_children[v]:
                return topo_index[c]
            ptr += 1
        return None
    
    for v in topo_order:
        if time.time() > deadline:
            break
        
        parents_list = parents.get(v, [])
        
        # Load red parents
        for p in parents_list:
            if state[p] == 'red':
                operations.append(['load', p])
                state[p] = 'blue'
                fast_mem.add(p)
        
        # Evict if over budget
        while len(fast_mem) > fast_memory_budget:
            candidates = [u for u in fast_mem if u != v and u not in output_set]
            if not candidates:
                break
            
            # Prefer delete (no transfer cost) for vertices with no more uses
            deletable = [u for u in candidates if unprocessed_child_count[u] == 0]
            if deletable:
                victim = deletable[0]
                operations.append(['delete', victim])
                state[victim] = 'red'
                fast_mem.discard(victim)
            else:
                # Store the one with farthest next use
                best_victim = None
                best_dist = -1
                for u in candidates:
                    nu = get_next_use(u)
                    if nu is None:
                        # Should have been in deletable
                        dist = float('inf')
                    else:
                        dist = nu - topo_index[v]
                    if dist > best_dist:
                        best_dist = dist
                        best_victim = u
                if best_victim is not None:
                    operations.append(['store', best_victim])
                    state[best_victim] = 'red'
                    fast_mem.discard(best_victim)
                else:
                    break
        
        # Compute v if needed
        if state[v] == 'red':
            operations.append(['compute', v])
            state[v] = 'blue'
            fast_mem.add(v)
        
        # Update parent bookkeeping
        for p in parents_list:
            processed_children[p].add(v)
            unprocessed_child_count[p] -= 1
        
        # Proactively delete vertices that are no longer needed
        # to free up space and reduce future transfer costs
        for u in list(fast_mem):
            if u in output_set or u == v:
                continue
            if unprocessed_child_count[u] == 0:
                operations.append(['delete', u])
                state[u] = 'red'
                fast_mem.discard(u)
    
    # Ensure outputs are blue
    for v in output_set:
        if state[v] == 'red':
            operations.append(['load', v])
            state[v] = 'blue'
            fast_mem.add(v)
        while len(fast_mem) > fast_memory_budget:
            candidates = [u for u in fast_mem if u not in output_set]
            if not candidates:
                break
            deletable = [u for u in candidates if unprocessed_child_count[u] == 0]
            if deletable:
                victim = deletable[0]
                operations.append(['delete', victim])
            else:
                victim = candidates[0]
                operations.append(['store', victim])
            state[victim] = 'red'
            fast_mem.discard(victim)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
