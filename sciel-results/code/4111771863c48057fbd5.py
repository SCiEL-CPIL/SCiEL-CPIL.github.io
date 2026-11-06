import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Try Paley Graphs
    for q in [37, 29, 17]:
        if time.time() > deadline: break
        graph = get_paley_graph(q)
        if is_valid(graph):
            return graph
            
    # 2. Try Simulated Annealing for n=35
    # ...
    
    # 3. Return Seed
    return get_paley_graph(17)
