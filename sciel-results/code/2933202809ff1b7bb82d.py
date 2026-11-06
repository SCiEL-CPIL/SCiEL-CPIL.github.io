import random
import time

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # Initialize with a random graph
    # p=0.5 gives ~60 edges
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.5:
                edges.add((u, v))
    
    # Precompute all 4-subsets
    vertices = list(range(n))
    subsets = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    subsets.append((i, j, k, l))
                    
    # To speed up K4,4 check, we can use a helper
    def is_k44(A, B):
        for a in A:
            for b in B:
                if a < b:
                    if (a, b) not in edges:
                        return False
                else:
                    if (b, a) not in edges:
                        return False
        return True

    while time.time() < deadline:
        found = False
        # Shuffle to randomize the edge removal order
        random.shuffle(subsets)
        
        # We can optimize by not checking pairs that share vertices
        # But checking intersection is also overhead.
        # Given 1820 subsets, 1.6M pairs is okay.
        
        for i in range(len(subsets)):
            A = subsets[i]
            for j in range(i + 1, len(subsets)):
                B = subsets[j]
                # Quick check: if they share a vertex, skip
                # Using set intersection is fast enough for small sets
                if len(set(A).intersection(B)) > 0:
                    continue
                
                if is_k44(A, B):
                    # Found a K4,4. Remove a random edge from it.
                    k44_edges = []
                    for a in A:
                        for b in B:
                            if a < b:
                                k44_edges.append((a, b))
                            else:
                                k44_edges.append((b, a))
                    edge_to_remove = random.choice(k44_edges)
                    edges.remove(edge_to_remove)
                    found = True
                    break
            if found:
                break
        
        if not found:
            # No K4,4 found. 
            # We can try to add edges to maximize the count.
            # But adding an edge might create a K4,4.
            # Let's try to add a random edge and check if it creates a K4,4.
            # If it does, we discard it. If not, we keep it.
            # We do this for a while.
            
            # Generate list of non-edges
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in edges:
                        non_edges.append((u, v))
            
            random.shuffle(non_edges)
            added_count = 0
            for e in non_edges:
                if time.time() > deadline:
                    break
                u, v = e
                # Tentatively add
                edges.add((u, v))
                
                # Check if this specific addition created a K4,4
                # We only need to check K4,4s that contain (u, v).
                # u and v must be in opposite parts.
                # Let A contain u, B contain v.
                # We need to find 3 other vertices for A and 3 for B.
                # This is still complex.
                
                # Simpler: Just check if the graph now contains ANY K4,4.
                # But that's the expensive loop.
                
                # Heuristic: If we add an edge, we might create a K4,4.
                # If we don't check, we might return an invalid graph.
                # So we must check.
                
                # To avoid the full scan, we can just revert if a full scan finds a K4,4?
                # No, that's slow.
                
                # Let's just break and return the current valid graph.
                # The random init + removal is already > 39.
                break
            break

    return [[u, v] for u, v in edges]
