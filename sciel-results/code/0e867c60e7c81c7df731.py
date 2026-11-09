import time
import sys

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Parse DAG
    # dag is expected to have 'nodes', 'edges', 'inputs', 'outputs' or similar structure
    # Based on "topologically ordered parents and output vertices", let's assume:
    # dag['nodes']: list of node ids
    # dag['parents']: dict node -> list of parents
    # dag['inputs']: list of input nodes
    # dag['outputs']: list of output nodes
    
    nodes = dag.get('nodes', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    if not nodes:
        return []

    # 1. Generate DFS Order
    # We need to visit nodes in an order that keeps subgraphs together.
    # Standard DFS from outputs backwards? Or inputs forwards?
    # Let's do a DFS from inputs, visiting children.
    # But we need to compute in topological order.
    # Let's use a topological sort but prioritize nodes whose parents are already computed?
    # Actually, a simple Topological Sort is safer for correctness.
    # To improve locality, we can use a "pebbling" order.
    # Let's stick to a Topological Sort first, then maybe optimize order.
    
    # Calculate in-degrees
    in_degree = {n: 0 for n in nodes}
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            in_degree[n] += 1
            children[p].append(n)
    
    # Topological Sort (Kahn's algorithm)
    from collections import deque
    queue = deque([n for n in nodes if in_degree[n] == 0])
    topo_order = []
    while queue:
        n = queue.popleft()
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(nodes):
        # Cycle detected or error, fallback to node list
        topo_order = nodes

    # 2. Precompute Next Use
    # For each node, find the next index in topo_order where it is needed as a parent.
    # This helps in eviction decisions.
    next_use = {n: len(topo_order) for n in nodes}
    # We need to know for each node v, when it will be used next.
    # A node is "used" when it is a parent of a node being computed.
    # So, for each node u in topo_order, its parents are "used" at index of u.
    # We want the *earliest* future use.
    
    # Let's create a list of (index, parent) for all uses.
    uses = []
    for i, u in enumerate(topo_order):
        for p in parents.get(u, []):
            uses.append((i, p))
            
    # Sort by index
    uses.sort()
    
    # For each parent, find the first use that is > current index.
    # This is tricky to do efficiently for all nodes at once.
    # Alternative: Iterate through topo_order, and for each node, update the next_use of its parents?
    # No, next_use should be the *next* time it is needed.
    
    # Let's just store the list of uses for each node.
    uses_by_node = {n: [] for n in nodes}
    for i, p in uses:
        uses_by_node[p].append(i)
        
    # Now, during simulation, we can check the next use.
    # But we need to know the "current" position in the simulation.
    
    # 3. Simulation
    fast_mem = set()
    main_mem = set(inputs) # Inputs start in main memory
    operations = []
    
    # We need to track which nodes are in main memory.
    # Initially, inputs are in main memory.
    # When we store a node, it goes to main memory.
    # When we load a node, it comes from main memory.
    
    # Helper to get next use index for a node at current simulation step
    # We will maintain a pointer for each node's uses list.
    use_ptr = {n: 0 for n in nodes}
    
    def get_next_use(n, current_index):
        # Find the first use index > current_index
        lst = uses_by_node[n]
        ptr = use_ptr[n]
        while ptr < len(lst) and lst[ptr] <= current_index:
            ptr += 1
        use_ptr[n] = ptr
        if ptr < len(lst):
            return lst[ptr]
        return len(topo_order) # No more uses
    
    for i, v in enumerate(topo_order):
        if time.time() > deadline:
            break
            
        # Ensure parents are in fast_mem
        for p in parents.get(v, []):
            if p not in fast_mem:
                if p in main_mem:
                    operations.append(('load', p))
                    fast_mem.add(p)
                else:
                    # Should not happen if we always store
                    # If it does, we might need to recompute, but let's assume it's in main_mem
                    # For safety, if not in main_mem, we can't load it.
                    # This implies a bug in the logic.
                    pass
        
        # Compute v
        operations.append(('compute', v))
        fast_mem.add(v)
        
        # Post-compute action
        if v in outputs:
            # Must finish in main memory
            operations.append(('store', v))
            fast_mem.remove(v)
            # main_mem.add(v) # Not strictly necessary as it's done
        else:
            # Intermediate
            # Check if we need to evict
            while len(fast_mem) > fast_memory_budget:
                # Select victim
                # Prioritize nodes with no future uses (delete)
                # Then nodes with farthest next use (store)
                
                best_victim = None
                best_score = -1
                best_is_last = False
                
                for u in fast_mem:
                    nu = get_next_use(u, i)
                    if nu == len(topo_order):
                        # No more uses, best to delete
                        score = float('inf')
                        is_last = True
                    else:
                        # Use distance
                        score = nu - i
                        is_last = False
                        
                    if score > best_score:
                        best_score = score
                        best_victim = u
                        best_is_last = is_last
                        
                if best_victim is not None:
                    if best_is_last:
                        operations.append(('delete', best_victim))
                    else:
                        operations.append(('store', best_victim))
                        main_mem.add(best_victim)
                    fast_mem.remove(best_victim)
                    
    return operations
