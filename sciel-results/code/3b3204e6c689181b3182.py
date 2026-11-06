"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    def get_vertices(d):
        if hasattr(d, 'vertices'):
            return list(d.vertices)
        elif isinstance(d, dict) and 'vertices' in d:
            return list(d['vertices'])
        elif hasattr(d, 'nodes'):
            return list(d.nodes)
        elif isinstance(d, dict) and 'nodes' in d:
            return list(d['nodes'])
        else:
            if hasattr(d, 'parents'):
                parents = d.parents
            elif isinstance(d, dict) and 'parents' in d:
                parents = d['parents']
            else:
                raise ValueError("Cannot determine vertices from dag")
            verts = set()
            for v, ps in parents.items():
                verts.add(v)
                verts.update(ps)
            return list(verts)
    
    def get_parents(d):
        if hasattr(d, 'parents'):
            return dict(d.parents)
        elif isinstance(d, dict) and 'parents' in d:
            return dict(d['parents'])
        elif hasattr(d, 'edges'):
            parents = {}
            for e in d.edges:
                if len(e) == 2:
                    child, parent = e
                    parents.setdefault(child, []).append(parent)
            return parents
        else:
            raise ValueError("Cannot determine parents from dag")
    
    def get_inputs(d):
        if hasattr(d, 'inputs'):
            return list(d.inputs)
        elif isinstance(d, dict) and 'inputs' in d:
            return list(d['inputs'])
        else:
            parents = get_parents(d)
            return [v for v in get_vertices(d) if not parents.get(v, [])]
    
    def get_outputs(d):
        if hasattr(d, 'outputs'):
            return list(d.outputs)
        elif isinstance(d, dict) and 'outputs' in d:
            return list(d['outputs'])
        else:
            parents = get_parents(d)
            vertices = get_vertices(d)
            children = {}
            for v, ps in parents.items():
                for p in ps:
                    children.setdefault(p, []).append(v)
            return [v for v in vertices if not children.get(v, [])]
    
    try:
        vertices = get_vertices(dag)
        parents_map = get_parents(dag)
        inputs = get_inputs(dag)
        outputs = get_outputs(dag)
    except Exception:
        # Fallback to a simple valid schedule if parsing fails badly, 
        # though we assume parsing works based on previous scores.
        # If we can't parse, we can't do better than baseline, but we must return something.
        # Since we can't import baseline easily without knowing if it exists in all contexts, 
        # we try a very naive fallback.
        try:
            # Try to extract minimal info
            if hasattr(dag, 'vertices'): verts = list(dag.vertices)
            elif isinstance(dag, dict) and 'vertices' in dag: verts = list(dag['vertices'])
            else: return []
            
            if hasattr(dag, 'parents'): pmap = dict(dag.parents)
            elif isinstance(dag, dict) and 'parents' in dag: pmap = dict(dag['parents'])
            else: pmap = {v: [] for v in verts}
            
            if hasattr(dag, 'inputs'): ins = set(dag.inputs)
            elif isinstance(dag, dict) and 'inputs' in dag: ins = set(dag['inputs'])
            else: ins = set(v for v in verts if not pmap.get(v, []))
            
            if hasattr(dag, 'outputs'): outs = set(dag.outputs)
            elif isinstance(dag, dict) and 'outputs' in dag: outs = set(dag['outputs'])
            else:
                children = defaultdict(list)
                for v, ps in pmap.items():
                    for p in ps: children[p].append(v)
                outs = set(v for v in verts if not children.get(v, []))
            
            # Naive topo sort
            in_degree = {v: len(pmap.get(v, [])) for v in verts}
            q = deque([v for v in verts if in_degree[v] == 0])
            topo = []
            while q:
                v = q.popleft()
                topo.append(v)
                for c in children.get(v, []):
                    in_degree[c] -= 1
                    if in_degree[c] == 0: q.append(c)
            
            ops = []
            mem = set()
            for v in topo:
                if v in ins:
                    if v not in mem:
                        if len(mem) >= fast_memory_budget:
                            # Evict LRU (just pick first)
                            to_evict = next(iter(mem))
                            ops.append(['store', to_evict])
                            mem.remove(to_evict)
                        ops.append(['load', v])
                        mem.add(v)
                    continue
                for p in pmap.get(v, []):
                    if p not in mem:
                        if len(mem) >= fast_memory_budget:
                            to_evict = next(iter(mem))
                            ops.append(['store', to_evict])
                            mem.remove(to_evict)
                        ops.append(['load', p])
                        mem.add(p)
                ops.append(['compute', v])
                mem.add(v)
            
            for v in outs:
                if v in mem:
                    ops.append(['store', v])
                    mem.remove(v)
            return ops
        except Exception:
            return []

    # Build children map
    children_map = defaultdict(list)
    for v, ps in parents_map.items():
        for p in ps:
            children_map[p].append(v)
            
    # Topological sort
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(vertices):
        topo_order = list(vertices)

    input_set = set(inputs)
    output_set = set(outputs)
    
    # Compute depth (layer) for each vertex
    # Depth 0: inputs (no parents)
    # Depth d: max(depth(parents)) + 1
    depth = {}
    for v in topo_order:
        if v in input_set:
            depth[v] = 0
        else:
            ps = parents_map.get(v, [])
            if not ps:
                depth[v] = 0 # Fallback
            else:
                depth[v] = max(depth.get(p, 0) for p in ps) + 1
                
    max_depth = max(depth.values()) if depth else 0
    layers = defaultdict(list)
    for v in vertices:
        layers[depth[v]].append(v)
        
    # Sort vertices within each layer by topological order to ensure determinism
    pos_in_topo = {v: i for i, v in enumerate(topo_order)}
    for d in layers:
        layers[d].sort(key=lambda x: pos_in_topo[x])

    fast_mem = set()
    operations = []
    op_count = 0
    MAX_OPS = 20000
    transfer_count = 0
    
    # Helper to evict if full
    def ensure_space():
        nonlocal op_count, transfer_count
        if len(fast_mem) < fast_memory_budget:
            return
        # Find a vertex to evict. 
        # Strategy: Evict a vertex that is not an output and is not needed in the current or next layer.
        # If all are needed, evict the one with the farthest next use.
        # For simplicity and speed, we just evict an arbitrary non-output if possible.
        # A better heuristic: Evict the vertex with the highest "last use" layer.
        
        # Pre-compute last use layer for each vertex? 
        # last_use_layer[v] = max depth of children of v, or -1 if none.
        # If last_use_layer[v] < current_layer, it's dead.
        
        # We'll compute "last_needed_layer" on the fly or precompute.
        pass

    # Precompute for each vertex: 
    # 1. Is it an output?
    # 2. What is the max depth of its children? (This tells us when it's no longer needed)
    last_needed_layer = {}
    for v in vertices:
        children = children_map.get(v, [])
        if not children:
            # If no children, it's a sink. If it's an output, we need it until the end.
            # If it's not an output, it's never needed after it's computed? 
            # Actually, if it's not an output and has no children, it's useless? 
            # But the task says "all outputs must finish blue". Non-outputs don't need to be blue at the end.
            # So non-output sinks can be deleted immediately after computation if they aren't used elsewhere.
            # But wait, if it's a sink, it has no children, so it's not used by anything.
            # So last_needed_layer is effectively its own depth.
            last_needed_layer[v] = depth[v]
        else:
            # It is needed until the last child is processed.
            # The last child is at max depth of children.
            last_needed_layer[v] = max(depth[c] for c in children)
            
    # We process layer by layer.
    # At the start of layer d, we need all parents of vertices in layer d to be in fast_mem.
    # After processing layer d, we can delete vertices whose last_needed_layer < d+1 (i.e., not needed in d+1 or later).
    # Note: If a vertex is needed in layer d+1, we must keep it.
    
    for d in range(max_depth + 1):
        if time.time() > deadline or op_count >= MAX_OPS:
            break
            
        current_layer = layers.get(d, [])
        if not current_layer:
            continue
            
        # 1. Ensure all parents of vertices in current_layer are in fast_mem
        # Collect all unique parents needed for this layer
        needed_parents = set()
        for v in current_layer:
            for p in parents_map.get(v, []):
                if p not in fast_mem:
                    needed_parents.add(p)
                    
        # Load needed parents
        # Sort parents by their last_needed_layer to prioritize keeping those needed longer?
        # Or just load them. If budget is tight, we might need to evict.
        
        # Optimization: If we have space, just load.
        # If not, we need to evict.
        # Who to evict? 
        # We should evict vertices that are NOT needed in the current layer or future layers.
        # i.e., last_needed_layer[v] < d.
        # If all vertices in fast_mem are needed now, we have a problem. 
        # But in a valid DAG pebbling, if we process in topological order, 
        # we should only need to keep things that are active.
        
        # Let's identify "dead" vertices in fast_mem: those with last_needed_layer < d
        dead_vertices = [v for v in fast_mem if last_needed_layer[v] < d]
        
        # If we have dead vertices, delete them to free space (no transfer cost for delete? 
        # The prompt says "delete" is an operation. Usually delete is free or cheap, 
        # but here "transfer_count" tracks load/store. Delete is likely free of transfer cost but counts as op.
        # The score is "mean transfers". So delete is good for freeing space without transfer penalty.
        # Let's delete all dead vertices first.
        for v in dead_vertices:
            if op_count >= MAX_OPS: break
            operations.append(['delete', v])
            fast_mem.remove(v)
            op_count += 1
            
        # Now, load the needed parents
        for p in needed_parents:
            if time.time() > deadline or op_count >= MAX_OPS:
                break
            if p not in fast_mem:
                # Check space
                if len(fast_mem) < fast_memory_budget:
                    operations.append(['load', p])
                    fast_mem.add(p)
                    op_count += 1
                    transfer_count += 1
                else:
                    # Need to evict. 
                    # Find a vertex in fast_mem that is NOT needed in the current layer (d) or later.
                    # i.e., last_needed_layer[v] < d.
                    # We already deleted those above. 
                    # So if we are still full, it means ALL vertices in fast_mem are needed in layer d or later.
                    # This implies we don't have enough budget to hold all active values + new parents.
                    # In this case, we must evict a value that is needed later, causing a re-load.
                    # Heuristic: Evict the one with the furthest last_needed_layer (furthest future need)
                    # OR evict the one with the smallest last_needed_layer (closest future need) to minimize re-load distance?
                    # Actually, if we evict a value needed in d+1, we have to load it again in d+1.
                    # If we evict a value needed in d+10, we have to load it again in d+10.
                    # It's usually better to evict the one with the *shortest* remaining horizon? 
                    # No, if we evict the one with the *longest* horizon, we keep the short ones which are needed soon.
                    # Let's evict the one with the MAX last_needed_layer (furthest need) to keep the "hot" values.
                    
                    evict_candidate = None
                    max_last_need = -1
                    for v in fast_mem:
                        if last_needed_layer[v] > max_last_need:
                            max_last_need = last_needed_layer[v]
                            evict_candidate = v
                            
                    if evict_candidate:
                        operations.append(['store', evict_candidate]) # Store to disk, so we can reload later
                        fast_mem.remove(evict_candidate)
                        op_count += 1
                        transfer_count += 1
                        
                        operations.append(['load', p])
                        fast_mem.add(p)
                        op_count += 1
                        transfer_count += 1
                    else:
                        # Should not happen if logic is correct
                        break

        # 2. Compute vertices in current layer
        for v in current_layer:
            if time.time() > deadline or op_count >= MAX_OPS:
                break
            if v in input_set:
                # Inputs are already loaded if they were parents of something, 
                # or if they are in the current layer and needed, they were loaded above.
                # But what if an input is in the current layer but NOT a parent of anything in this layer?
                # It might be a parent of a later layer. It was loaded when it was needed.
                # If it's an input and it's in the current layer, it doesn't need computation.
                # But does it need to be in fast_mem? 
                # If it's not a parent of anything in the current layer, it wasn't loaded above.
                # It will be loaded when its first child is processed.
                # So we just skip computation.
                continue
                
            # Check if v is already in fast_mem? 
            # It shouldn't be, unless it was loaded as a parent? 
            # No, v is in layer d. Its parents are in layers < d.
            # v itself is new.
            
            # Ensure space for v
            if len(fast_mem) >= fast_memory_budget:
                # Evict something
                # Same logic: evict furthest need
                evict_candidate = None
                max_last_need = -1
                for v2 in fast_mem:
                    if last_needed_layer[v2] > max_last_need:
                        max_last_need = last_needed_layer[v2]
                        evict_candidate = v2
                if evict_candidate:
                    operations.append(['store', evict_candidate])
                    fast_mem.remove(evict_candidate)
                    op_count += 1
                    transfer_count += 1

            operations.append(['compute', v])
            fast_mem.add(v)
            op_count += 1
            
        # 3. Cleanup: Delete vertices that are no longer needed
        # A vertex v is no longer needed if last_needed_layer[v] < d+1
        # i.e., last_needed_layer[v] <= d
        # Note: If v is an output, we must NOT delete it. We must store it at the end.
        # But we can't delete outputs.
        
        for v in list(fast_mem):
            if op_count >= MAX_OPS: break
            if v in output_set:
                continue
            if last_needed_layer[v] <= d:
                operations.append(['delete', v])
                fast_mem.remove(v)
                op_count += 1
                
    # 4. Finalize: Store all outputs
    for v in outputs:
        if time.time() > deadline or op_count >= MAX_OPS:
            break
        if v in fast_mem:
            operations.append(['store', v])
            fast_mem.remove(v)
            op_count += 1
            transfer_count += 1
        else:
            # Output is not in fast_mem. 
            # It must be in slow memory (blue). 
            # If it was computed and stored earlier, it's blue.
            # If it was never loaded/computed? That's an error in our schedule.
            # But if it's an input that is also an output, it starts blue.
            # So we don't need to do anything.
            pass
            
    return operations[:MAX_OPS]

# EVOLVE-BLOCK-END
