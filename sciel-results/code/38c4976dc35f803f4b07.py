import numpy as np

def priority(adjacency):
        n = adjacency.shape[0]
        active = np.ones(n, dtype=bool)
        scores = np.zeros(n, dtype=float)
        
        for step in range(n):
            if not np.any(active):
                break
            
            # Calculate degrees in the active subgraph
            deg = (adjacency @ active).astype(int)
            
            # Mask inactive nodes so they are not selected
            # We want to maximize degree, so set inactive to -1
            deg[~active] = -1
            
            # Select vertex with maximum degree
            v = np.argmax(deg)
            
            # Assign score: higher score means picked earlier by the greedy decoder
            # The directive says: "assign scores proportional to the layer in which they were removed"
            # and "ensuring high-priority vertices are those that would be selected"
            # So step 0 (first selected) should have highest score.
            scores[v] = n - step
            
            # Remove v and its neighbors from the active set
            active[v] = False
            active[adjacency[v]] = False
            
        return scores
