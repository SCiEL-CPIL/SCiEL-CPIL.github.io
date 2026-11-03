import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    def generate_random_regular(n, k):
        # Configuration model
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        edges = set()
        adj = [[] for _ in range(n)]
        
        # Pair stubs
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Loop
                # Find another stub to swap with
                # To keep it simple, we might just restart if this happens too often
                # But for n=320, k=4, loops are rare?
                # P(loop) = 1/n.
                # Expected loops = n*k/2 * 1/n = k/2 = 2.
                # So we expect 2 loops.
                # We need to fix them.
                pass
            
            if (u, v) in edges or (v, u) in edges:
                # Multi-edge
                pass
                
        # This simple generation is flawed.
        # Let's use a different method:
        # 1. Start with a cycle 0-1-...-n-1-0.
        # 2. Add random chords to increase degree to 4.
        #    Each vertex currently has degree 2.
        #    We need to add 2 more edges per vertex.
        #    Total 2n edges to add.
        #    We can pick random pairs of "half-edges" and connect them.
        
        adj = [[] for _ in range(n)]
        # Base cycle
        for i in range(n):
            u = i
            v = (i + 1) % n
            adj[u].append(v)
            adj[v].append(u)
            
        # Remaining stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * (k - 2)) # k=4, so 2 stubs per node
            
        random.shuffle(stubs)
        
        # Connect stubs
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            # Check if edge exists
            if v in adj[u]:
                # Multi-edge, skip or swap?
                # For simplicity, we might accept multi-edges in intermediate steps
                # and fix them later? No, must be simple.
                # Let's just retry if multi-edge?
                # Or swap with another stub.
                pass
            adj[u].append(v)
            adj[v].append(u)
            
        # Convert to edge list
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_set.add((u, v))
                elif u > v:
                    edge_set.add((v, u))
        return list(edge_set), adj

    # The above generator is still risky.
    # I will provide a robust solution using a simple random restart and local search.
    
    pass
