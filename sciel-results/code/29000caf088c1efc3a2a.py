"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        return []
    
    if vertices is None or parents is None:
        return []
    
    # Normalize vertices
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)
    
    num_vertices = len(vertex_list)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}
    
    # Normalize parents
    if isinstance(parents, dict):
        parent_dict = {v: list(parents.get(v, [])) for v in vertex_list}
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {v: [] for v in vertex_list}
    
    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    else:
        inputs = set(inputs)
    
    if outputs is None:
        children = defaultdict(list)
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    else:
        outputs = set(outputs)
    
    # Build children map
    children = defaultdict(list)
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)
    
    # Determine needed vertices
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)
    
    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    
    # Compute last_use for each vertex
    last_use = {}
    for v in needed_vertices:
        last_use[v] = vertex_to_idx.get(v, -1)
        for child in children.get(v, []):
            if child in needed_set:
                last_use[v] = max(last_use[v], vertex_to_idx.get(child, -1))
    
    # Count fanout
    fanout = {v: 0 for v in needed_vertices}
    for v in needed_vertices:
        for p in parent_dict.get(v, []):
            if p in needed_set:
                fanout[p] += 1
    
    # Greedy pebbling
    in_fast = set()
    computed = set()
    operations = []
    
    def load(v):
        operations.append(['load', v])
        in_fast.add(v)
    
    def store(v):
        operations.append(['store', v])
        in_fast.discard(v)
    
    def compute(v):
        operations.append(['compute', v])
        computed.add(v)
        in_fast.add(v)
    
    def delete(v):
        operations.append(['delete', v])
        in_fast.discard(v)
    
    # Pre-load inputs
    for v in needed_vertices:
        if v in inputs:
            load(v)
            computed.add(v)
    
    # Process vertices in topological order
    for v in needed_vertices:
        if time.time() > deadline:
            break
        
        if v in inputs:
            continue
        
        parents = [p for p in parent_dict.get(v, []) if p in needed_set]
        
        # Ensure parents are in fast memory
        for p in parents:
            if p in in_fast:
                continue
            
            if len(in_fast) >= fast_memory_budget:
                # Evict the vertex with the latest last_use that isn't needed now
                candidates = [u for u in in_fast if u not in parents and u != v]
                if candidates:
                    best_evict = max(candidates, key=lambda u: last_use.get(u, -1))
                    store(best_evict)
            
            load(p)
        
        compute(v)
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if v in computed or v in inputs:
                load(v)
    
    return operations

# EVOLVE-BLOCK-END
