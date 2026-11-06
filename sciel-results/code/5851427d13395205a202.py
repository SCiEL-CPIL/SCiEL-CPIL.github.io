"""Communication-minimizing DAG pebbling - from-scratch design."""
# EVOLVE-BLOCK-START

import sys
from collections import deque

def _topo_layers(dag):
    """Infer layered structure from dag dict.
    dag format assumed: dict with 'vertices' (list), 'parents' (list of lists), 'outputs' (list), 'inputs' (list).
    Vertices are topologically ordered: parents[i] are indices < i.
    """
    vertices = dag.get('vertices', dag.get('V', []))
    parents = dag.get('parents', dag.get('P', []))
    outputs = dag.get('outputs', dag.get('O', []))
    inputs = dag.get('inputs', dag.get('I', []))
    
    n = len(vertices) if vertices else len(parents)
    if not parents and n > 0:
        parents = [[] for _ in range(n)]
    
    # Compute in-degree (number of children) for topological processing
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
    
    # Layer assignment: longest path from an input
    layer = [0] * n
    # Inputs have layer 0
    for i in inputs:
        layer[i] = 0
    
    # Topological order is given (parents before children)
    for i in range(n):
        for c in children[i]:
            if layer[c] < layer[i] + 1:
                layer[c] = layer[i] + 1
    
    return n, parents, children, inputs, outputs, layer

def _naive_schedule(dag, fast_memory_budget):
    """Naive schedule: compute in topological order, load parents, compute, store result if needed later,
    delete when no longer needed. Count transfers."""
    n, parents, children, inputs, outputs, layer = _topo_layers(dag)
    
    if n == 0:
        return 0
    
    # Track which vertices are still needed (have unprocessed children)
    remaining_children = [len(children[i]) for i in range(n)]
    
    transfers = 0
    fast_set = set()
    
    # Initialize: inputs are blue (in fast memory)
    for i in inputs:
        fast_set.add(i)
    
    # Process vertices in topological order
    for v in range(n):
        # Ensure all parents are in fast memory (load if not)
        for p in parents[v]:
            if p not in fast_set:
                # Load p
                transfers += 1
                fast_set.add(p)
        
        # Compute v
        # After computing, v is in fast memory
        fast_set.add(v)
        
        # If fast memory is over budget, evict something
        while len(fast_set) > fast_memory_budget:
            # Find a vertex that is no longer needed (no remaining children) and not an output
            evicted = False
            for u in list(fast_set):
                if u not in inputs and u not in outputs and remaining_children[u] == 0:
                    fast_set.discard(u)
                    transfers += 1  # store
                    evicted = True
                    break
            if not evicted:
                # Evict any non-input, non-output
                for u in list(fast_set):
                    if u not in inputs and u not in outputs:
                        fast_set.discard(u)
                        transfers += 1
                        evicted = True
                        break
                if not evicted:
                    # Should not happen if budget >= 1
                    break
        
        # Update remaining children
        for c in children[v]:
            remaining_children[c] -= 1
    
    # At the end, all outputs must be blue. They should be if we kept them.
    # But let's verify and load if needed
    for o in outputs:
        if o not in fast_set:
            transfers += 1
            fast_set.add(o)
    
    return transfers

def _optimal_pebbling(dag, fast_memory_budget):
    """
    Advanced pebbling using a cost-optimal eviction strategy.
    
    Key idea: We process vertices in topological order. At each step, we decide which
    vertices to keep in fast memory. The cost is the number of transfers (loads + stores).
    
    We use a greedy strategy with lookahead:
    - Keep vertices that will be reused soon (high reuse frequency)
    - Evict vertices that won't be reused or will be reused far in the future
    """
    n, parents, children, inputs, outputs, layer = _topo_layers(dag)
    
    if n == 0:
        return 0
    
    # Precompute: for each vertex, the list of children that need it
    # and the "next use" distance
    
    # remaining_children[v] = number of children of v not yet processed
    remaining_children = [len(children[i]) for i in range(n)]
    
    # For each vertex v, compute the "urgency": how soon it will be needed again
    # We process in topological order, so we can precompute the next child index
    
    # Build a priority-based eviction: evict the vertex with the farthest next use
    # or the one that is no longer needed
    
    transfers = 0
    fast_set = set()
    
    # Initialize with inputs
    for i in inputs:
        fast_set.add(i)
    
    # Track which vertices have been computed
    computed = [False] * n
    
    # For each vertex, we track the list of children that still need it
    # We use a heap for fast eviction decisions
    
    # Precompute for each vertex v, the sorted list of children
    # and for each child, its position in topological order
    
    # For eviction priority: we want to evict the vertex that is "least needed"
    # Score: if a vertex has no more children, it can be evicted (unless it's an output)
    # If it has children, we estimate the cost of evicting it: we'll need to reload it
    # for each remaining child, but we can reload once and keep it
    
    # Better approach: use a replacement policy based on "distance to next use"
    
    # Let's use a simpler but effective strategy:
    # - When we need to evict, evict the vertex with the maximum "next use distance"
    # - "next use distance" = distance to the next child that needs it, or infinity if none
    
    # Precompute next_use[v] = the minimum topological index of a child of v not yet processed
    # This is dynamic, so we'll maintain it
    
    # Actually, let's use a different approach: cost-based scheduling
    
    # For each vertex v, define:
    # - in_degree = len(parents[v])
    # - out_degree = len(children[v])
    # - is_input = v in inputs
    # - is_output = v in outputs
    
    # The naive schedule loads each parent for each child. We can optimize by keeping
    # vertices in fast memory if they will be reused.
    
    # Let's use a dynamic programming approach on the layered structure.
    
    # For a layered DAG, we can think of it as a sequence of layers.
    # At each layer, we compute all vertices in that layer.
    # We need to decide which vertices from previous layers to keep.
    
    # Group vertices by layer
    max_layer = max(layer) if layer else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in range(n):
        layers[layer[v]].append(v)
    
    # Process layer by layer
    fast_set = set()
    transfers = 0
    
    # Initialize with inputs (layer 0)
    for i in inputs:
        fast_set.add(i)
    
    # For each layer, we compute all vertices in that layer
    # Before computing, ensure all parents are in fast memory
    
    for L in range(max_layer + 1):
        # For each vertex in this layer, ensure parents are loaded
        for v in layers[L]:
            for p in parents[v]:
                if p not in fast_set:
                    transfers += 1
                    fast_set.add(p)
        
        # Compute all vertices in this layer
        for v in layers[L]:
            fast_set.add(v)
        
        # Now we may be over budget. We need to evict some vertices.
        # We should evict vertices that:
        # 1. Are not outputs
        # 2. Have no more children in future layers
        # 3. Among those, evict the ones with the farthest next use
        
        # Compute for each vertex in fast_set, the next use (next child layer)
        # A vertex v is needed in future if it has children in layers > L
        
        while len(fast_set) > fast_memory_budget:
            # Find the best vertex to evict
            best_to_evict = None
            best_score = -1
            
            for u in fast_set:
                # Don't evict inputs (they're always blue? No, they start blue but can be evicted)
                # Actually, inputs start blue, but they can be evicted and reloaded
                # Don't evict outputs (they must finish blue)
                if u in outputs:
                    continue
                
                # Check if u has any children in future layers
                has_future_child = False
                for c in children[u]:
                    if layer[c] > L:
                        has_future_child = True
                        break
                
                if not has_future_child:
                    # Can evict safely (no more uses)
                    # Score: prefer to evict vertices with more "slack"
                    # All such vertices are equally good, pick any
                    best_to_evict = u
                    best_score = 1000
                    break
                else:
                    # Has future children, evicting means we'll reload later
                    # Score: evict the one with the farthest next use
                    # (so we can keep it for closer uses)
                    next_use_layer = min(layer[c] for c in children[u] if layer[c] > L)
                    # Farther next use = lower priority to keep = higher priority to evict
                    score = next_use_layer - L
                    if score > best_score:
                        best_score = score
                        best_to_evict = u
            
            if best_to_evict is None:
                # No vertex can be evicted (all are outputs or have future children)
                # This means the budget is too small. We have to evict something.
                # Evict the vertex with the farthest next use, even if it has future children
                for u in fast_set:
                    if u in outputs:
                        continue
                    next_use_layer = min((layer[c] for c in children[u] if layer[c] > L), default=max_layer + 1)
                    score = next_use_layer - L
                    if score > best_score:
                        best_score = score
                        best_to_evict = u
                
                if best_to_evict is None:
                    # All remaining are outputs, can't evict
                    break
            
            fast_set.discard(best_to_evict)
            transfers += 1  # store
    
    # At the end, ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast_set:
            transfers += 1
            fast_set.add(o)
    
    return transfers

def _advanced_pebbling(dag, fast_memory_budget):
    """
    Even more advanced: use a cost model to decide the optimal eviction policy.
    
    For each vertex v, define:
    - cost_keep(v) = 0 if we keep v in fast memory
    - cost_evict(v) = number of times v will need to be reloaded in the future
    
    We use a greedy strategy: at each eviction decision, evict the vertex with
    the highest cost_evict / cost_keep ratio, i.e., the one where evicting is
    most expensive relative to keeping.
    
    Actually, we want to minimize total transfers, so we should evict the vertex
    where the marginal cost of evicting is lowest.
    """
    n, parents, children, inputs, outputs, layer = _topo_layers(dag)
    
    if n == 0:
        return 0
    
    max_layer = max(layer) if layer else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in range(n):
        layers[layer[v]].append(v)
    
    fast_set = set()
    transfers = 0
    
    for i in inputs:
        fast_set.add(i)
    
    # For each vertex, precompute the number of future children (by layer)
    # future_children[v][L] = number of children of v in layer >= L
    future_children = {}
    for v in range(n):
        fc = [0] * (max_layer + 2)
        for c in children[v]:
            for L in range(layer[c], max_layer + 1):
                fc[L] += 1
        future_children[v] = fc
    
    for L in range(max_layer + 1):
        # Ensure parents are loaded
        for v in layers[L]:
            for p in parents[v]:
                if p not in fast_set:
                    transfers += 1
                    fast_set.add(p)
        
        # Compute vertices in this layer
        for v in layers[L]:
            fast_set.add(v)
        
        # Evict if over budget
        while len(fast_set) > fast_memory_budget:
            best_to_evict = None
            best_cost = float('inf')
            
            for u in fast_set:
                if u in outputs:
                    continue
                
                # Cost of evicting u: number of future reloads
                # If u has k future children, evicting means we need to reload u
                # before processing each of those children (unless we reload and keep)
                # Actually, we can reload u once and keep it, so the cost is 1 store + 1 load
                # But if we have multiple future children, we might reload multiple times
                # if we evict and then need it again later
                
                # Simplified: cost = number of future children that will require a reload
                # If we evict u, the next time we need u, we load it. If we keep it,
                # we don't need to load it again.
                
                # The number of future "uses" of u is the number of children in future layers.
                # But we can reload u once and use it for all remaining children.
                # So the cost of evicting is: 1 store (now) + 1 load (when next needed)
                # = 2, regardless of the number of future children.
                
                # However, if u is needed in multiple disjoint "clusters" of children,
                # we might need to reload multiple times. But since we process in topological
                # order, all children of u are in layers > L, and they are processed in order.
                # So we can reload u once and keep it for all remaining children.
                
                # So the cost of evicting u is approximately 2 (store + load) if u has
                # at least one future child, or 1 (store) if u has no future children.
                
                num_future = future_children[u][L + 1] if L + 1 <= max_layer else 0
                
                if num_future == 0:
                    cost = 1  # just store, no reload needed
                else:
                    cost = 2  # store now, load later
                
                # But we also need to consider: if we evict u, we free up one slot.
                # The benefit is that we can keep another vertex that has more future uses.
                # So we should evict the vertex with the lowest "value" of keeping.
                
                # Value of keeping u = number of future uses (children in future layers)
                # We should evict the vertex with the lowest value.
                
                value = num_future
                
                # We want to evict the vertex with the lowest value, breaking ties by
                # preferring to evict vertices that are "closer" to being unused.
                
                if value < best_cost or (value == best_cost and best_to_evict is not None and layer[u] < layer[best_to_evict]):
                    # Wait, this is getting complicated. Let's just evict the one with
                    # the fewest future children.
                    best_cost = value
                    best_to_evict = u
            
            if best_to_evict is None:
                # No vertex can be evicted (all are outputs)
                break
            
            fast_set.discard(best_to_evict)
            transfers += 1
    
    # Ensure outputs are in fast memory
    for o in outputs:
        if o not in fast_set:
            transfers += 1
            fast_set.add(o)
    
    return transfers

def pebble(dag, fast_memory_budget):
    """
    Main entry point. Returns a list of [operation, vertex] operations.
    Operations: 'load', 'store', 'compute', 'delete'
    
    We generate a schedule that minimizes transfers.
    """
    n, parents, children, inputs, outputs, layer = _topo_layers(dag)
    
    if n == 0:
        return []
    
    max_layer = max(layer) if layer else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in range(n):
        layers[layer[v]].append(v)
    
    # We'll generate a schedule using the advanced pebbling strategy
    # and count the transfers
    
    fast_set = set()
    operations = []
    transfers = 0
    
    for i in inputs:
        fast_set.add(i)
    
    # Precompute future children counts
    future_children = {}
    for v in range(n):
        fc = [0] * (max_layer + 2)
        for c in children[v]:
            for L in range(layer[c], max_layer + 1):
                fc[L] += 1
        future_children[v] = fc
    
    for L in range(max_layer + 1):
        # Load parents if needed
        for v in layers[L]:
            for p in parents[v]:
                if p not in fast_set:
                    operations.append(['load', p])
                    fast_set.add(p)
                    transfers += 1
        
        # Compute vertices in this layer
        for v in layers[L]:
            operations.append(['compute', v])
            fast_set.add(v)
        
        # Evict if over budget
        while len(fast_set) > fast_memory_budget:
            best_to_evict = None
            best_value = float('inf')
            
            for u in fast_set:
                if u in outputs:
                    continue
                
                num_future = future_children[u][L + 1] if L + 1 <= max_layer else 0
                
                # Value of keeping u = number of future uses
                value = num_future
                
                if value < best_value:
                    best_value = value
                    best_to_evict = u
            
            if best_to_evict is None:
                break
            
            operations.append(['store', best_to_evict])
            fast_set.discard(best_to_evict)
            transfers += 1
    
    # Ensure outputs are in fast memory
    for o in outputs:
        if o not in fast_set:
            operations.append(['load', o])
            fast_set.add(o)
            transfers += 1
    
    return operations

# EVOLVE-BLOCK-END
