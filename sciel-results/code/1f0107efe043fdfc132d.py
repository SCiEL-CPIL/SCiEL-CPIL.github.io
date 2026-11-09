import numpy
import time
import heapq
from collections import defaultdict
import numpy as np

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    fast_memory_budget = int(fast_memory_budget)
    
    def normalize_node(n):
        # Try to convert to int
        try:
            # Handle numpy types and python numerics
            if isinstance(n, (int, float, np.integer, np.floating)):
                return int(n)
            # Handle strings that represent numbers
            if isinstance(n, str):
                return int(n)
            # Handle other types that might be convertible
            return int(n)
        except (ValueError, TypeError):
            # If it cannot be converted to int, we have a problem.
            # For safety, we might need to map it later, but let's return as is for now
            # or raise an error to see what it is.
            # Given the error "vertex must be an integer", it's highly likely
            # these are convertible.
            return n

    # --- Parse DAG Structure ---
    # We need to handle dag being a dict, list, or numpy array.
    
    dag_dict = {}
    outputs = None
    
    if isinstance(dag, dict):
        # Check if 'outputs' is a key
        if 'outputs' in dag:
            outputs = [normalize_node(o) for o in dag['outputs']]
            for k, v in dag.items():
                if k != 'outputs':
                    node = normalize_node(k)
                    parents = [normalize_node(p) for p in v]
                    dag_dict[node] = parents
        else:
            for k, v in dag.items():
                node = normalize_node(k)
                parents = [normalize_node(p) for p in v]
                dag_dict[node] = parents
            # Infer outputs
            all_nodes = set(dag_dict.keys())
            parent_set = set()
            for parents in dag_dict.values():
                parent_set.update(parents)
            outputs = list(all_nodes - parent_set)
            
    elif isinstance(dag, (list, tuple, np.ndarray)):
        # Assume list of [node, parents]
        for item in dag:
            if isinstance(item, (list, tuple, np.ndarray)):
                if len(item) >= 2:
                    node = normalize_node(item[0])
                    parents = [normalize_node(p) for p in item[1]]
                    dag_dict[node] = parents
                elif len(item) == 1:
                    node = normalize_node(item[0])
                    dag_dict[node] = []
            else:
                # Fallback: maybe just node IDs?
                node = normalize_node(item)
                dag_dict[node] = []
        
        # Infer outputs
        all_nodes = set(dag_dict.keys())
        parent_set = set()
        for parents in dag_dict.values():
            parent_set.update(parents)
        outputs = list(all_nodes - parent_set)
    else:
        return []

    # Ensure all nodes in dag_dict and outputs are integers
    # If any are not, we need to remap.
    all_unique_nodes = set(dag_dict.keys())
    for p_list in dag_dict.values():
        all_unique_nodes.update(p_list)
    if outputs:
        all_unique_nodes.update(outputs)
        
    non_int_nodes = [n for n in all_unique_nodes if not isinstance(n, int)]
    
    if non_int_nodes:
        # Remap all nodes to 0..N-1
        node_map = {}
        idx = 0
        for n in sorted(all_unique_nodes, key=lambda x: (type(x).__name__, x)):
            node_map[n] = idx
            idx += 1
            
        new_dag_dict = {}
        for node, parents in dag_dict.items():
            new_node = node_map[node]
            new_parents = [node_map[p] for p in parents]
            new_dag_dict[new_node] = new_parents
            
        if outputs:
            outputs = [node_map[o] for o in outputs]
            
        dag_dict = new_dag_dict
        
    # Build children map
    children = defaultdict(list)
    for node, parents in dag_dict.items():
        for p in parents:
            children[p].append(node)
    
    # Identify Inputs (nodes with no parents)
    inputs = [n for n, p in dag_dict.items() if not p]
    
    # Topological order
    in_degree = {n: len(p) for n, p in dag_dict.items()}
    queue = [n for n in dag_dict if in_degree[n] == 0]
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        n = queue.pop(0)
        topo_order.append(n)
        for c in children[n]:
            temp_in_degree[c] -= 1
            if temp_in_degree[c] == 0:
                queue.append(c)
                
    # Descendant count
    desc_count = {n: 0 for n in dag_dict}
    for n in reversed(topo_order):
        if children[n]:
            desc_count[n] = 1 + sum(desc_count[c] for c in children[n])
        else:
            desc_count[n] = 0
            
    # State
    blue_set = set(inputs)
    red_set = set(dag_dict.keys()) - blue_set
    empty_set = set()
    
    operations = []
    
    if len(blue_set) > fast_memory_budget:
        sorted_inputs = sorted(inputs, key=lambda x: desc_count[x])
        while len(blue_set) > fast_memory_budget:
            victim = sorted_inputs.pop(0)
            if victim in blue_set:
                blue_set.remove(victim)
                red_set.add(victim)
                operations.append(['store', victim])
                
    max_ops = 20000
    op_count = 0
    
    while op_count < max_ops:
        if time.time() - start_time > 50:
            break
            
        if all(o in blue_set for o in outputs):
            break
            
        ready_nodes = []
        for node in dag_dict:
            if node in red_set or node in empty_set:
                if all(p in blue_set for p in dag_dict[node]):
                    ready_nodes.append(node)
                    
        if ready_nodes:
            ready_nodes.sort(key=lambda x: desc_count[x], reverse=True)
            v = ready_nodes[0]
            
            if len(blue_set) >= fast_memory_budget:
                candidates = [n for n in blue_set if n not in outputs]
                if not candidates:
                    candidates = list(blue_set)
                    
                candidates.sort(key=lambda x: desc_count[x])
                victim = candidates[0]
                
                blue_set.remove(victim)
                red_set.add(victim)
                operations.append(['store', victim])
                op_count += 1
                if op_count >= max_ops:
                    break
                
            blue_set.add(v)
            red_set.discard(v)
            operations.append(['compute', v])
            op_count += 1
            
        else:
            candidates = []
            for node in red_set:
                for c in children[node]:
                    if c in red_set or c in empty_set:
                        almost_ready = True
                        for p in dag_dict[c]:
                            if p != node and p not in blue_set:
                                almost_ready = False
                                break
                        if almost_ready:
                            candidates.append((node, desc_count[c]))
                            break
            
            if not candidates:
                candidates = [(n, desc_count[n]) for n in red_set]
                if not candidates:
                    break
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            u = candidates[0][0]
            
            if len(blue_set) >= fast_memory_budget:
                candidates_evict = [n for n in blue_set if n not in outputs]
                if not candidates_evict:
                    candidates_evict = list(blue_set)
                candidates_evict.sort(key=lambda x: desc_count[x])
                victim = candidates_evict[0]
                blue_set.remove(victim)
                red_set.add(victim)
                operations.append(['store', victim])
                op_count += 1
                if op_count >= max_ops:
                    break
                
            blue_set.add(u)
            red_set.discard(u)
            operations.append(['load', u])
            op_count += 1

    return operations
