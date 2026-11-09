"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = set()
    
    # Hubs: 0 and 20
    # Rim 1: 1..19
    # Rim 2: 21..39
    
    # Spokes from 0 to Rim 1
    for i in range(1, 20):
        edges.add((0, i))
        
    # Spokes from 20 to Rim 2
    for i in range(21, 40):
        edges.add((20, i))
        
    # Path in Rim 1: 1-2-...-19
    for i in range(1, 19):
        edges.add((i, i+1))
        
    # Path in Rim 2: 21-22-...-39
    for i in range(21, 39):
        edges.add((i, i+1))
        
    # Hub connection
    edges.add((0, 20))
    
    # Bridge between Rims (18 to 21)
    edges.add((18, 21))
    
    edge_list = list(edges)
    
    cycles = []
    
    # Cycles of length 3 to 20 (using Hub 0 and Rim 1)
    for k in range(3, 21):
        # Cycle: 0, 1, ..., k-1, 0
        cyc = list(range(1, k))
        cyc.insert(0, 0)
        cycles.append(cyc)
        
    # Cycles of length 21 to 40 (using Hub 0, Rim 1, Bridge, Rim 2, Hub 20)
    for k in range(21, 41):
        if k == 21:
            # 0, 1, ..., 18, 21, 20, 0
            cyc = [0] + list(range(1, 19)) + [21, 20]
        elif k == 40:
            # 0, 1, ..., 19, 18, 21, ..., 39, 20, 0
            # Note: 19 is included, then back to 18 to cross bridge
            cyc = [0] + list(range(1, 20)) + [18] + list(range(21, 40)) + [20]
        else:
            # General case:
            # We need k vertices.
            # Base: 0, 1..18 (19 vertices), 21 (1), 20 (1) -> 21 vertices.
            # We need to add (k - 21) vertices from Rim 2 (22..39).
            # Or add vertices from Rim 1?
            # My previous analysis:
            # k=21: 0, 1..18, 21, 20. (19 + 1 + 1 + 1 = 22? No. 1+18+1+1=21).
            # Wait, range(1, 19) is 1..18. Count 18.
            # 0 (1) + 18 + 21 (1) + 20 (1) = 21. Correct.
            
            # For k > 21, we extend into Rim 2.
            # Add vertices 22, 23, ... up to m.
            # Count added: (k - 21).
            # So we include 21, 22, ..., 21 + (k - 21) = k.
            # Wait, max index in Rim 2 is 39.
            # If k=22, add 22. Vertices: 0, 1..18, 21, 22, 20.
            # Path: 0-1-...-18-21-22-20-0.
            
            # For k=40, we need all vertices.
            # 0, 1..19, 18, 21..39, 20.
            # This uses 19 from Rim 1.
            
            # Let's construct generally:
            # If k <= 21 + (39 - 21) = 39?
            # Max extension in Rim 2 is 39.
            # Base 21 + 18 (22..39) = 39.
            # So for k=39: 0, 1..18, 21..39, 20.
            # For k=40: We must include 19 as well.
            
            if k <= 39:
                # Use 1..18 from Rim 1
                # Use 21..(21 + k - 21) = k from Rim 2?
                # Vertices: 0, 1..18, 21..k, 20.
                # Count: 1 + 18 + (k - 20) + 1 = k.
                # Range for Rim 2 part: 21 to k.
                # Edge from 18 to 21 exists.
                # Edge from k to 20 exists (spoke).
                cyc = [0] + list(range(1, 19)) + list(range(21, k+1)) + [20]
            else:
                # k=40
                # Include 19 in Rim 1.
                # Path: 0-1-...-19-18-21-...-39-20-0
                cyc = [0] + list(range(1, 20)) + [18] + list(range(21, 40)) + [20]
                
        cycles.append(cyc)
        
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
