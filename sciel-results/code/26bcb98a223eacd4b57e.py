import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin

    # 1. Try Paley(41)
    if time.time() < deadline:
        n, edges = build_paley(41)
        if is_valid(n, edges):
            return n, edges

    # 2. Start with Paley(37)
    n, edges = build_paley(37)
    if not is_valid(n, edges):
        return 0, [] # Should not happen

    # 3. Try to extend
    adj = build_adj(n, edges)
    max_extend = 5
    for _ in range(max_extend):
        if time.time() > deadline - 10:
            break
        new_adj, success = try_extend(adj, deadline)
        if success:
            adj = new_adj
            n += 1
        else:
            break
            
    edges = adj_to_edges(adj)
    return n, edges
