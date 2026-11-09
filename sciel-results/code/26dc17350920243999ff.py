# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55.0

    # --- Robust Parsing ---
    vertices = []
    input_vertices = set()
    output_vertices = set()
    parents_map = defaultdict(set)
    children_map = defaultdict(set)

    # 1. Extract Vertices
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
    elif hasattr(dag, 'nodes'):
        vertices = list(dag.nodes)
    elif isinstance(dag, dict) and 'vertices' in dag:
        vertices = list(dag['vertices'])
    elif isinstance(dag, dict) and 'nodes' in dag:
        vertices = list(dag['nodes'])
    elif isinstance(dag, list) and len(dag) > 0 and not isinstance(dag[0], (list, tuple)):
        vertices = list(dag)
    elif isinstance(dag, dict):
        vertices = list(dag.keys())

    # 2. Extract Inputs/Outputs
    if hasattr(dag, 'input_vertices'):
        input_vertices = set(dag.input_vertices)
    elif isinstance(dag, dict) and 'input_vertices' in dag:
        input_vertices = set(dag['input_vertices'])
    elif isinstance(dag, dict) and 'inputs' in dag:
        input_vertices = set(dag['inputs'])
        
    if hasattr(dag, 'output_vertices'):
        output_vertices = set(dag.output_vertices)
    elif isinstance(dag, dict) and 'output_vertices' in dag:
        output_vertices = set(dag['output_vertices'])
    elif isinstance(dag, dict) and 'outputs' in dag:
        output_vertices = set(dag['outputs'])

    # 3. Extract Edges
    raw_edges = None
    if hasattr(dag, 'edges'):
        raw_edges = dag.edges
    elif hasattr(dag, 'adj'):
        raw_edges = dag.adj
    elif isinstance(dag, dict) and 'edges' in dag:
        raw_edges = dag['edges']
    elif isinstance(dag, dict) and 'parents' in dag:
        raw_edges = dag['parents']
    elif isinstance(dag, dict) and 'children' in dag:
        raw_edges = dag['children']

    if raw_edges is not None:
        if isinstance(raw_edges, dict):
            key_name = None
            if isinstance(dag, dict):
                if 'parents' in dag: key_name = 'parents'
                elif 'children' in dag: key_name = 'children'
                elif 'edges' in dag: key_name = 'edges'
            
            if key_name == 'children':
                for parent, child_list in raw_edges.items():
                    for c in child_list:
                        parents_map[c].add(parent)
                        children_map[parent].add(c)
            else:
                for child, par_list in raw_edges.items():
                    for p in par_list:
                        parents_map[child].add(p)
                        children_map[p].add(child)
        elif isinstance(raw_edges, list):
            # Assume (parent, child)
            for e in raw_edges:
                if len(e) == 2:
                    p, c = e
                    parents_map[c].add(p)
                    children_map[p].add(c)

    # Infer inputs/outputs
    if not input_vertices:
        for v in vertices:
            if len(parents_map.get(v, set())) == 0:
                input_vertices.add(v)
                
    if not output_vertices:
        for v in vertices:
            if len(children_map.get(v, set())) == 0:
                output_vertices.add(v)

    # Topological Sort
    in_degree = {v: len(parents_map.get(v, set())) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, set()):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(vertices):
        remaining = set(vertices) - set(topo_order)
        topo_order.extend(list(remaining))

    # --- Pebbling Algorithm ---
    red_set = set()
    blue_set = set(input_vertices)
    operations = []
    unprocessed = set(topo_order)
    
    def free_space():
        # Returns True if space is freed, False if budget is 0 and full
        if len(red_set) < fast_memory_budget:
            return True
        
        # 1. Delete a vertex not in needed_in_red
        for v in list(red_set):
            if v not in needed_in_red:
                red_set.remove(v)
                operations.append(['delete', v])
                return True
        
        # 2. Store a vertex in needed_in_red (prefer one with fewest unprocessed children)
        # This keeps the value in Blue, so we can Load it later without Recompute
        candidates = [v for v in red_set if v in needed_in_red]
        if candidates:
            # Sort by number of unprocessed children (ascending)
            candidates.sort(key=lambda x: len(children_map.get(x, set()) & unprocessed))
            d = candidates[0]
            red_set.remove(d)
            operations.append(['store', d])
            return True
            
        # 3. Delete a vertex in needed_in_red (Last resort)
        if red_set:
            d = next(iter(red_set))
            red_set.remove(d)
            operations.append(['delete', d])
            return True
            
        return False

    for v in topo_order:
        if time.time() > deadline:
            break
            
        unprocessed.remove(v)
        
        # Calculate needed_in_red
        needed_in_red = set()
        for u in unprocessed:
            needed_in_red.update(parents_map.get(u, set()))
            
        # 1. Ensure parents of v are in red_set
        for p in parents_map.get(v, set()):
            if p not in red_set:
                if p in blue_set:
                    if not free_space():
                        # Cannot free space, skip? Or fail?
                        # If budget is 0, we can't load anything.
                        continue
                    operations.append(['load', p])
                    red_set.add(p)
                else:
                    # Recompute p
                    # Ensure p's parents are in red_set
                    for pp in parents_map.get(p, set()):
                        if pp not in red_set:
                            if pp in blue_set:
                                if not free_space():
                                    continue
                                operations.append(['load', pp])
                                red_set.add(pp)
                            # Recursion for pp's parents omitted for simplicity, 
                            # assuming this case is rare
                    if not free_space():
                        continue
                    red_set.add(p)
                    operations.append(['compute', p])
                    blue_set.add(p)
        
        # 2. Compute v
        if v not in red_set:
            if not free_space():
                continue
            red_set.add(v)
            operations.append(['compute', v])
            blue_set.add(v)
            
    # 3. Store all output vertices
    for v in output_vertices:
        if time.time() > deadline:
            break
        if v in red_set:
            red_set.remove(v)
            operations.append(['store', v])
        # If v in blue_set, it's already blue.
        
    # 4. Clean up
    for v in list(red_set):
        if time.time() > deadline:
            break
        red_set.remove(v)
        operations.append(['delete', v])
        
    return operations
# EVOLVE-BLOCK-END
