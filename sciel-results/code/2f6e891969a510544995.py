import numpy as np
import heapq
import time

def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Precompute neighbors as lists for efficiency
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    original_degrees = adjacency.sum(axis=1).astype(int)
    
    def peel_min_degree():
        """Standard min-degree peeling (core number heuristic)."""
        if time.time() > deadline:
            return None
        current_degrees = original_degrees.copy()
        removed = np.zeros(n, dtype=bool)
        removal_order = np.zeros(n, dtype=int)
        step = 0
        heap = [(current_degrees[i], i) for i in range(n)]
        heapq.heapify(heap)
        
        while heap:
            if time.time() > deadline:
                return None
            d, u = heapq.heappop(heap)
            if removed[u]:
                continue
            if d != current_degrees[u]:
                continue
            removed[u] = True
            removal_order[u] = step
            step += 1
            for v in neighbors[u]:
                if not removed[v]:
                    current_degrees[v] -= 1
                    heapq.heappush(heap, (current_degrees[v], v))
        return removal_order
    
    def peel_max_neighbor_degree():
        """Prefer vertices whose neighbors have high total degree (dominator heuristic)."""
        if time.time() > deadline:
            return None
        # Score: prefer low own degree, but among ties, prefer high sum of neighbor degrees
        # Actually, for MIS we want to pick vertices that "block" many others, so high degree is bad
        # But we also want to pick vertices that are "central" in a sense
        # Let's use: score = own_degree - sum_of_neighbor_degrees (lower is better to pick first)
        # Wait, for greedy MIS, we want to pick vertices that have few neighbors (so we don't block many)
        # But we also want to pick vertices that are not "covered" by other potential picks
        # Let's try: prefer vertices with low degree, tie-break by high neighbor degree sum
        # (vertices adjacent to high-degree nodes are more "valuable" to include early)
        
        neighbor_degree_sum = np.zeros(n, dtype=int)
        for i in range(n):
            if len(neighbors[i]) > 0:
                neighbor_degree_sum[i] = original_degrees[neighbors[i]].sum()
        
        # Primary: low degree. Secondary: high neighbor degree sum
        # We want to pick low-degree vertices first, but if two have same degree,
        # pick the one whose neighbors are higher-degree (more "central" in some sense)
        # Actually, let's reverse: pick vertices that have few neighbors AND whose neighbors
        # also have few neighbors (peripheral vertices)
        # Let's use: score = degree + neighbor_degree_sum (lower is better)
        scores = original_degrees.astype(float) + neighbor_degree_sum.astype(float)
        
        removed = np.zeros(n, dtype=bool)
        removal_order = np.zeros(n, dtype=int)
        step = 0
        # Use a different tie-breaking: sort by (score, vertex_id)
        order = np.argsort(scores, kind='stable')
        
        # But we need dynamic peeling, not static order
        # Let's do dynamic: at each step, pick the vertex with min (current_degree, -neighbor_degree_sum_of_remaining)
        # This is complex. Let's just use a static order based on initial scores
        # and then simulate the greedy selection
        # Actually, for the ensemble, let's just generate a static order and use it
        
        # For consistency with the peeling approach, let's do a proper dynamic peel
        # with a different priority function
        current_degrees = original_degrees.copy()
        # We'll use a heap with (current_degree, -neighbor_degree_sum_initial, vertex)
        heap = [(current_degrees[i], -neighbor_degree_sum[i], i) for i in range(n)]
        heapq.heapify(heap)
        
        while heap:
            if time.time() > deadline:
                return None
            d, neg_nds, u = heapq.heappop(heap)
            if removed[u]:
                continue
            if d != current_degrees[u]:
                continue
            removed[u] = True
            removal_order[u] = step
            step += 1
            for v in neighbors[u]:
                if not removed[v]:
                    current_degrees[v] -= 1
                    heapq.heappush(heap, (current_degrees[v], -neighbor_degree_sum[v], v))
        return removal_order
    
    def peel_randomized_min_degree():
        """Min-degree peeling with random tie-breaking among equal degrees."""
        if time.time() > deadline:
            return None
        current_degrees = original_degrees.copy()
        removed = np.zeros(n, dtype=bool)
        removal_order = np.zeros(n, dtype=int)
        step = 0
        heap = [(current_degrees[i], np.random.random(), i) for i in range(n)]
        heapq.heapify(heap)
        
        while heap:
            if time.time() > deadline:
                return None
            d, rnd, u = heapq.heappop(heap)
            if removed[u]:
                continue
            if d != current_degrees[u]:
                continue
            removed[u] = True
            removal_order[u] = step
            step += 1
            for v in neighbors[u]:
                if not removed[v]:
                    current_degrees[v] -= 1
                    heapq.heappush(heap, (current_degrees[v], np.random.random(), v))
        return removal_order
    
    # Run multiple peeling orders
    orders = []
    
    # Order 1: Standard min-degree
    order1 = peel_min_degree()
    if order1 is not None:
        orders.append(order1)
    
    # Order 2: Max neighbor degree heuristic
    order2 = peel_max_neighbor_degree()
    if order2 is not None:
        orders.append(order2)
    
    # Order 3: Randomized min-degree
    order3 = peel_randomized_min_degree()
    if order3 is not None:
        orders.append(order3)
    
    if not orders:
        # Fallback: just use degree-based scores
        return -original_degrees.astype(float)
    
    # Combine orders into a consensus score
    # For each vertex, compute its average rank across all orders
    # Lower average rank = higher priority
    avg_rank = np.zeros(n, dtype=float)
    for order in orders:
        avg_rank += order.astype(float)
    avg_rank /= len(orders)
    
    # Score: lower average rank should give higher score
    # Also incorporate original degree as a tie-breaker
    scores = -avg_rank * (n + 1) - original_degrees.astype(float)
    
    return scores
