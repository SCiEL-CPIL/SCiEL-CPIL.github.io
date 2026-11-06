# EVOLVE-BLOCK-START
import time
from collections import deque

def _extract(dag):
    """Extract vertex list, parents dict, outputs, inputs from a flexible dag object."""
    verts = getattr(dag, "vertices", None)
    if verts is None:
        verts = getattr(dag, "nodes", None)
    if verts is None:
        verts = getattr(dag, "V", None)
    if verts is None and isinstance(dag, dict):
        verts = list(dag.keys())
    if verts is None:
        raise ValueError("cannot find vertices in dag")
    verts = list(verts)

    parents = {}
    for v in verts:
        p = None
        if isinstance(dag, dict) and v in dag and isinstance(dag[v], dict):
            p = dag[v].get("parents")
        if p is None:
            p = getattr(dag, "parents", None)
            if callable(p):
                p = p(v)
        if p is None:
            p = getattr(dag, "parent_map", {}).get(v, [])
        if p is None:
            p = []
        parents[v] = list(p)

    outputs = getattr(dag, "outputs", None)
    if outputs is None:
        outputs = getattr(dag, "sinks", None)
    if outputs is None and isinstance(dag, dict):
        outputs = dag.get("outputs", [])
    if outputs is None:
        outputs = []
    outputs = list(outputs)

    inputs = [v for v in verts if len(parents[v]) == 0]

    indeg = {v: len(parents[v]) for v in verts}
    q = deque(v for v in verts if indeg[v] == 0)
    topo = []
    child = {v: [] for v in verts}
    for v in verts:
        for p in parents[v]:
            child[p].append(v)
    while q:
        u = q.popleft()
        topo.append(u)
        for w in child[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(verts):
        topo = verts
    return verts, parents, outputs, inputs, topo, child

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    try:
        verts, parents, outputs, inputs, topo, child = _extract(dag)
    except Exception:
        return []

    if fast_memory_budget is None or fast_memory_budget < 1:
        fast_memory_budget = 1
    budget = int(fast_memory_budget)

    input_set = set(inputs)
    output_set = set(outputs)
    
    ops = []
    in_mem = set()
    computed = set()

    # Initialize need counts for ALL vertices
    # need[v] = number of future computations that require v in memory
    need = {v: 0 for v in verts}
    
    # Mark outputs as needing to stay until end (conceptually need 1 more at the very end)
    # We handle outputs by forcing them to stay in memory at the end.
    # For intermediate scheduling, need[v] counts children not yet computed.
    for v in topo:
        if v in input_set:
            continue
        for p in parents[v]:
            need[p] += 1
            
    # Add 1 to outputs so they are considered "needed" even if no children depend on them
    # This helps the eviction heuristic keep them around longer if possible, 
    # but we will explicitly force them in memory at the end.
    for o in outputs:
        need[o] += 1

    # Initialize memory with inputs if possible
    # Inputs are "blue" at start. We can assume they are available.
    # Do they consume memory slots? "every red value consumes one fast-memory slot".
    # "Inputs start blue". Usually, blue means available in slow memory or external.
    # "load" brings it to fast memory (red).
    # If inputs don't consume slots until loaded, we don't add them to in_mem initially.
    # However, if we treat "blue" as "in fast memory", they do.
    # The prompt says: "every red value consumes one fast-memory slot".
    # "Inputs start blue". This implies they are NOT in fast memory initially?
    # Or does it mean they are available?
    # Standard pebbling: Inputs are white (available) or blue (in fast memory)?
    # Usually: White (uncomputed), Blue (in fast memory), Red (in slow memory).
    # Here: "Inputs start blue". If Blue = Fast Memory, they occupy slots.
    # Let's assume Blue = Fast Memory.
    # If so, we should add inputs to in_mem if budget allows.
    
    for v in inputs:
        if len(in_mem) < budget:
            in_mem.add(v)
            # Do we need a load op? No, it starts there.
        computed.add(v)
        
    # If budget is small, we might not fit all inputs. 
    # We will load them on demand.

    remaining = [v for v in topo if v not in input_set]
    
    def evict_one():
        # Find a vertex to evict.
        # Priority: 
        # 1. Not an output.
        # 2. Lowest need.
        # 3. If all are outputs or need is high, evict the one with lowest need anyway (recompute later).
        best = None
        min_need = float('inf')
        for v in in_mem:
            # We should never evict a vertex that is currently being computed? 
            # We call evict before compute, so v is not in in_mem yet.
            # We should not evict a vertex that is an output if possible?
            # But if budget is full and we need to load a parent for a non-output, 
            # and only outputs are in memory, we MUST evict an output (and reload it later).
            
            # Score: lower is better to evict.
            # Outputs are harder to evict (penalty).
            score = need[v]
            if v in output_set:
                score += 1000 # Strong penalty for evicting outputs
            
            if score < min_need:
                min_need = score
                best = v
                
        if best is not None:
            in_mem.discard(best)
            ops.append(["store", best])
            return best
        return None

    for v in remaining:
        if time.time() - t0 > 50:
            break
            
        # Ensure parents are in memory
        for p in parents[v]:
            if p not in in_mem:
                # Check if p is computed. It should be, since we process in topo order.
                if p not in computed:
                    continue # Should not happen in valid topo order
                
                # Make room
                while len(in_mem) >= budget and p not in in_mem:
                    evict_one()
                
                ops.append(["load", p])
                in_mem.add(p)
        
        # Compute v
        # Make room for v
        while len(in_mem) >= budget and v not in in_mem:
            evict_one()
            
        ops.append(["compute", v])
        in_mem.add(v)
        computed.add(v)
        
        # Update need for parents
        for p in parents[v]:
            need[p] -= 1

    # Ensure all outputs finish blue (in fast memory)
    for o in outputs:
        if o not in in_mem:
            if o not in computed:
                continue # Should not happen
            
            while len(in_mem) >= budget and o not in in_mem:
                evict_one()
            
            ops.append(["load", o])
            in_mem.add(o)

    if len(ops) > 20000:
        ops = ops[:20000]
    return ops
# EVOLVE-BLOCK-END
