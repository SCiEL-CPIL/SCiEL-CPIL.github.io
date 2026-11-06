# Precompute uses: for each vertex, the list of positions in topo_order where it's used
uses = {v: [] for v in vertices}
for i, v in enumerate(topo_order):
    for p in parent_map.get(v, []):
        uses[p].append(i)

# Process vertices in topological order
fast = set()
slow = set()
operations = []

# Load inputs
for v in inputs:
    if time.time() > deadline or len(operations) >= 20000:
        break
    operations.append(['load', v])
    fast.add(v)

for i, v in enumerate(topo_order):
    if time.time() > deadline or len(operations) >= 20000:
        break
    
    # Ensure all parents are in fast memory
    for p in parent_map.get(v, []):
        if p not in fast:
            if p in slow:
                operations.append(['load', p])
                fast.add(p)
                slow.remove(p)
            else:
                operations.append(['load', p])
                fast.add(p)
    
    # Compute v
    operations.append(['compute', v])
    fast.add(v)
    
    # Evict if over budget
    while len(fast) > fast_memory_budget:
        # Find the vertex in fast whose next use is farthest in the future
        best_v = None
        best_next = -1
        for u in fast:
            # Find the next use of u after position i
            next_pos = float('inf')
            for j in uses.get(u, []):
                if j > i:
                    next_pos = j
                    break
            if best_v is None or next_pos > best_next:
                best_v = u
                best_next = next_pos
        
        if best_v is None:
            best_v = next(iter(fast))
        
        # Store best_v
        operations.append(['store', best_v])
        fast.remove(best_v)
        slow.add(best_v)

# Store outputs
for v in outputs:
    if time.time() > deadline or len(operations) >= 20000:
        break
    if v in fast:
        operations.append(['store', v])
        fast.remove(v)
        slow.add(v)
    elif v not in slow:
        operations.append(['compute', v])
        fast.add(v)
        operations.append(['store', v])
        fast.remove(v)
        slow.add(v)

return operations
