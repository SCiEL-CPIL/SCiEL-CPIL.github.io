import numpy as np


def priority(adjacency):
        n = adjacency.shape[0]
        deg = adjacency.sum(axis=1).astype(int)
        
        # Compute Core Numbers
        # Simple O(N^2) or O(N+E) implementation
        # Since N=200, even O(N^2) is fine, but let's be efficient.
        core = np.zeros(n, dtype=int)
        remaining = np.ones(n, dtype=bool)
        current_deg = deg.copy()
        
        # Sort vertices by degree for processing?
        # Standard algorithm:
        # 1. Sort vertices by degree.
        # 2. Iterate. If current degree < current core threshold, remove.
        
        # Simpler approach for small N:
        # Repeatedly remove vertex with min degree, assign its degree as core number.
        # This is O(N^2) which is 40,000 ops. Trivial.
        
        order = np.argsort(deg)
        # We need to update degrees as we remove.
        # Let's use a queue or list.
        
        # Actually, the "peeling" algorithm:
        # 1. Count degrees.
        # 2. Put vertices with degree 0 in queue.
        # 3. While queue not empty:
        #      pop v. core[v] = degree[v].
        #      for u in neighbors(v):
        #           degree[u] -= 1
        #           if degree[u] < current_max_core? No.
        
        # Correct K-Core Algorithm:
        # 1. Compute degrees.
        # 2. Sort vertices by degree.
        # 3. Let k = 0.
        # 4. While vertices remain:
        #      Find vertex v with min degree.
        #      If deg(v) < k: 
        #          # This vertex belongs to k-1 core?
        #          # Actually, standard algorithm:
        #          # Iterate k from 0 to max_degree.
        
        # Let's use the simple peeling for N=200.
        # It's easier to just do:
        # core[v] = max k such that v is in k-core.
        
        # Efficient enough for N=200:
        # For k from 1 to max_deg:
        #    Check if there is a subgraph where all deg >= k.
        
        # Even simpler:
        # The core number of a vertex is the maximum k such that it has at least k neighbors with core number >= k?
        # No, that's recursive.
        
        # Let's implement the standard peeling algorithm.
        # 1. deg = adjacency.sum(axis=1)
        # 2. sorted_indices = argsort(deg)
        # 3. We maintain a list of active vertices.
        # 4. We iterate.
        
        # Actually, since N is small, we can just compute it directly.
        # core[v] is the largest k such that there exists a subgraph containing v where all degrees are >= k.
        
        # Let's use the algorithm from Batagelj and Zaversnik.
        # 1. Sort vertices by degree.
        # 2. Let P be the sorted list.
        # 3. Let i = 0.
        # 4. While i < n:
        #      Let v = P[i].
        #      If deg[v] < i: 
        #           # This vertex is removed?
        #           # Actually, the core number of v is the current "k" which is the min degree in the remaining graph?
        
        # Let's stick to a robust, simple method for N=200.
        # We can simulate the peeling.
        
        degs = deg.copy()
        core = np.zeros(n, dtype=int)
        active = np.ones(n, dtype=bool)
        
        # We need to pick the vertex with min degree among active vertices.
        # N=200, so we can just loop N times.
        for _ in range(n):
            # Find active vertex with min degree
            min_deg = np.inf
            v = -1
            for u in range(n):
                if active[u] and degs[u] < min_deg:
                    min_deg = degs[u]
                    v = u
            
            if v == -1:
                break
                
            core[v] = min_deg
            
            # Remove v
            active[v] = False
            for u in np.where(adjacency[v, :])[0]:
                if active[u]:
                    degs[u] -= 1
                    
        # Now we have core numbers.
        # We want to prioritize LOW core numbers.
        # And tie-break by LOW degree.
        
        # Score: -core * 1000 - degree
        scores = -core.astype(float) * 1000 - deg.astype(float)
        
        return scores
