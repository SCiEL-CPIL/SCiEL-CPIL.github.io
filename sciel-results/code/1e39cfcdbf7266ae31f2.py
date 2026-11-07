import itertools
import time

def construct():
    """
    Constructs two disjoint K3,3 graphs on 12 vertices.
    Cycle lengths in K3,3 are 4 and 6.
    Distinct cycle lengths count: 2.
    """
    # Time limit guard (though this function is O(1), it's good practice)
    deadline = time.time() + 59.0
    
    # Define bipartite sets for the first K3,3 component (vertices 0-5)
    set_a_1 = [0, 1, 2]
    set_b_1 = [3, 4, 5]
    
    # Define bipartite sets for the second K3,3 component (vertices 6-11)
    set_a_2 = [6, 7, 8]
    set_b_2 = [9, 10, 11]
    
    raw_edges = []
    
    # Generate edges for first component using Cartesian product
    for u, v in itertools.product(set_a_1, set_b_1):
        raw_edges.append((u, v))
        
    # Generate edges for second component using Cartesian product
    for u, v in itertools.product(set_a_2, set_b_2):
        raw_edges.append((u, v))
        
    # Convert to list of lists with strict integer types
    final_edges = []
    for u, v in raw_edges:
        final_edges.append([int(u), int(v)])
        
    return final_edges
