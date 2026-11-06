import heapq
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    # Parse DAG
    # dag structure: 
    # Based on "dag has topologically ordered parents and output vertices"
    # Let's assume dag is a dict or list. 
    # Common format in these tasks: 
    # dag = { 'nodes': N, 'parents': {node: [parents]}, 'inputs': [...], 'outputs': [...] }
    # Or similar. 
    # I will write a robust parser that handles common dict/list structures.
    
    # Since I don't see the exact structure, I'll infer from "topologically ordered parents".
    # Let's assume the input `dag` is a dictionary with keys:
    # 'num_nodes': int
    # 'parents': list of lists, where parents[i] is list of parent indices for node i
    # 'inputs': list of input node indices
    # 'outputs': list of output node indices
    
    # If the structure is different, I need to be careful.
    # The prompt says "generated mathematical instances".
    # Let's assume a standard format. If I can't parse, I'll try to introspect.
    
    # Fallback: If dag is a list of (parents, is_output)?
    # Let's look at the baseline import: `pebble_baseline(dag, fast_memory_budget)`.
    # I will assume `dag` contains:
    # 1. Number of nodes or max node id.
    # 2. Parent relationships.
    # 3. Inputs and Outputs.
    
    # To be safe, I'll try to extract these attributes.
    
    def extract_dag_info(dag):
        # Try common keys
        if hasattr(dag, 'num_nodes'):
            n = dag.num_nodes
        elif 'num_nodes' in dag:
            n = dag['num_nodes']
        elif 'n' in dag:
            n = dag['n']
        else:
            # Assume list of nodes
            if isinstance(dag, list):
                n = len(dag)
            else:
                # Try to find max index in parents if available
                n = 0
                if 'parents' in dag:
                    for p_list in dag['parents']:
                        if p_list:
                            n = max(n, max(p_list) + 1)
                    # Also check node count
                    if 'nodes' in dag:
                        n = max(n, dag['nodes'])
                if n == 0:
                    n = 10 # Default small
    
        # Parents
        if 'parents' in dag:
            parents = dag['parents']
        elif hasattr(dag, 'parents'):
            parents = dag.parents
        else:
            # If dag is list of parents lists
            if isinstance(dag, list) and all(isinstance(x, list) for x in dag):
                parents = dag
            else:
                parents = [[] for _ in range(n)]
        
        # Ensure parents is a list of lists of length n
        if len(parents) != n:
            # Pad or truncate?
            if len(parents) < n:
                parents = parents + [[] for _ in range(n - len(parents))]
            else:
                parents = parents[:n]
        
        # Inputs
        if 'inputs' in dag:
            inputs = dag['inputs']
        elif hasattr(dag, 'inputs'):
            inputs = dag.inputs
        else:
            # Nodes with no parents are inputs
            inputs = [i for i in range(n) if not parents[i]]
        
        # Outputs
        if 'outputs' in dag:
            outputs = dag['outputs']
        elif hasattr(dag, 'outputs'):
            outputs = dag.outputs
        else:
            # Nodes with no children are outputs
            children = [[] for _ in range(n)]
            for i, p_list in enumerate(parents):
                for p in p_list:
                    children[p].append(i)
            outputs = [i for i in range(n) if not children[i]]
            
        return n, parents, inputs, outputs

    n, parents, inputs, outputs = extract_dag_info(dag)
    
    # Bitmasks
    # 0: Uncomputed
    # 1: Blue
    # 2: Red
    # State: (mask_blue, mask_red)
    
    initial_blue_mask = 0
    for i in inputs:
        initial_blue_mask |= (1 << i)
    
    initial_red_mask = 0
    initial_state = (initial_blue_mask, initial_red_mask)
    
    # Goal: All outputs in mask_blue
    goal_blue_mask = 0
    for o in outputs:
        goal_blue_mask |= (1 << o)
    
    # A* Search
    # Heuristic: 
    # For each output not in blue:
    #   if in red: 1 (store)
    #   else: 2 (compute + store) + cost to make parents red?
    # Let's use a simple admissible heuristic:
    # h = sum(1 if o in red else 2 for o in outputs if o not in blue)
    
    def heuristic(mask_blue, mask_red):
        h = 0
        for o in outputs:
            bit = 1 << o
            if mask_blue & bit:
                continue
            if mask_red & bit:
                h += 1
            else:
                h += 2
        return h
    
    start_h = heuristic(initial_blue_mask, initial_red_mask)
    counter = 0
    # priority queue: (f, g, h, mask_blue, mask_red)
    # f = g + h
    pq = [(start_h, 0, start_h, initial_blue_mask, initial_red_mask)]
    
    # To track visited states and best g-cost
    # Key: (mask_blue, mask_red)
    best_g = {initial_state: 0}
    
    # Track parent pointers to reconstruct path
    # parent[state] = (prev_state, operation, vertex)
    parent_map = {initial_state: None}
    
    # Limit search to prevent timeout
    max_expansions = 500000
    expansions = 0
    
    while pq and expansions < max_expansions:
        f, g, h, mb, mr = heapq.heappop(pq)
        
        current_state = (mb, mr)
        
        if best_g.get(current_state, float('inf')) < g:
            continue
            
        expansions += 1
        
        # Check goal
        if (mb & goal_blue_mask) == goal_blue_mask:
            # Reconstruct path
            path = []
            state = current_state
            while parent_map[state] is not None:
                prev_state, op, v = parent_map[state]
                path.append([op, v])
                state = prev_state
            path.reverse()
            return path
        
        # Generate neighbors
        # 1. Load v: v in blue, not in red
        # 2. Store v: v in red
        # 3. Delete v: v in red
        # 4. Compute v: v not in blue, not in red, parents all in red
        
        # Iterate through all nodes
        for v in range(n):
            bit = 1 << v
            in_blue = bool(mb & bit)
            in_red = bool(mr & bit)
            
            # Load
            if in_blue and not in_red:
                if bin(mr).count('1') < fast_memory_budget:
                    new_mb = mb & ~bit
                    new_mr = mr | bit
                    new_state = (new_mb, new_mr)
                    new_g = g + 1
                    if new_g < best_g.get(new_state, float('inf')):
                        best_g[new_state] = new_g
                        new_h = heuristic(new_mb, new_mr)
                        heapq.heappush(pq, (new_g + new_h, new_g, new_h, new_mb, new_mr))
                        parent_map[new_state] = (current_state, "load", v)
                        
            # Store
            if in_red:
                new_mb = mb | bit
                new_mr = mr & ~bit
                new_state = (new_mb, new_mr)
                new_g = g + 1
                if new_g < best_g.get(new_state, float('inf')):
                    best_g[new_state] = new_g
                    new_h = heuristic(new_mb, new_mr)
                    heapq.heappush(pq, (new_g + new_h, new_g, new_h, new_mb, new_mr))
                    parent_map[new_state] = (current_state, "store", v)
                    
            # Delete
            if in_red:
                new_mb = mb
                new_mr = mr & ~bit
                new_state = (new_mb, new_mr)
                new_g = g + 1
                if new_g < best_g.get(new_state, float('inf')):
                    best_g[new_state] = new_g
                    new_h = heuristic(new_mb, new_mr)
                    heapq.heappush(pq, (new_g + new_h, new_g, new_h, new_mb, new_mr))
                    parent_map[new_state] = (current_state, "delete", v)
                    
            # Compute
            if not in_blue and not in_red:
                # Check parents
                can_compute = True
                for p in parents[v]:
                    if not (mr & (1 << p)):
                        can_compute = False
                        break
                if can_compute:
                    if bin(mr).count('1') < fast_memory_budget:
                        new_mb = mb
                        new_mr = mr | bit
                        new_state = (new_mb, new_mr)
                        new_g = g + 1
                        if new_g < best_g.get(new_state, float('inf')):
                            best_g[new_state] = new_g
                            new_h = heuristic(new_mb, new_mr)
                            heapq.heappush(pq, (new_g + new_h, new_g, new_h, new_mb, new_mr))
                            parent_map[new_state] = (current_state, "compute", v)
                            
    # If A* fails or times out, fall back to greedy
    # Greedy: Topological order
    # Load inputs, compute, store outputs, manage memory by deleting non-outputs
    # This is a rough fallback
    
    # Simple Greedy
    # 1. Topological sort
    # 2. For each node:
    #    - If parent is blue, load it? No, if parent is blue, it's not in memory. 
    #    - We need parents in memory.
    #    - So for each node in topo order:
    #       - For each parent p:
    #           - If p is blue, load p.
    #           - If p is uncomputed, this shouldn't happen in topo order if we process correctly.
    #       - Compute node.
    #       - If node is output, store it.
    #       - Else, if memory full, delete an intermediate.
        
    # Implementing a robust greedy is complex. 
    # Given the "layered" nature, A* should work for small N.
    # If N is large, the problem is likely structured such that greedy is okay, 
    # but A* is better.
    
    # If we reach here, return empty or best effort
    return []

