<<<<<<< SEARCH
    # --- Generate candidate graphs ---
    
    def make_paley_graph(q):
        """Create a Paley graph over GF(q) where q is a prime power, q ≡ 1 mod 4."""
        if q < 5 or q % 4 != 1:
            return None
        
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        if not is_prime(q):
            return None
        
        adj = np.zeros((q, q), dtype=bool)
        for i in range(q):
            for j in range(i + 1, q):
                diff = (j - i) % q
                if diff == 0:
                    continue
                if pow(diff, (q - 1) // 2, q) == 1:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj
    
    def make_random_graph(n, p, seed=42):
        """Create a random graph with edge probability p."""
        rng = np.random.default_rng(seed)
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj
    
    def make_complement(adj):
        """Return the complement of a graph."""
        n = adj.shape[0]
        comp = ~adj.copy()
        np.fill_diagonal(comp, False)
        return comp
=======
    # --- Generate candidate graphs ---
    
    def make_paley_graph(q):
        """Create a Paley graph over GF(q) where q is a prime power, q ≡ 1 mod 4."""
        if q < 5 or q % 4 != 1:
            return None
        
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        if not is_prime(q):
            return None
        
        adj = np.zeros((q, q), dtype=bool)
        for i in range(q):
            for j in range(i + 1, q):
                diff = (j - i) % q
                if diff == 0:
                    continue
                if pow(diff, (q - 1) // 2, q) == 1:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj
    
    def make_random_graph(n, p, seed=42):
        """Create a random graph with edge probability p."""
        rng = np.random.default_rng(seed)
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj
    
    def make_complement(adj):
        """Return the complement of a graph."""
        n = adj.shape[0]
        comp = ~adj.copy()
        np.fill_diagonal(comp, False)
        return comp
    
    def compute_irregularity_score(adj, n):
        """
        Compute a spectral-gap-based irregularity metric.
        For each 6-subset, compute the variance of induced degrees.
        Lower variance = more irregular = better.
        Returns the total sum of squared deviations (lower is better).
        Uses sampling for speed.
        """
        if n < 6:
            return 0.0
        
        A = adj.astype(np.int8)
        rng = np.random.default_rng(123)
        
        # Sample 6-subsets
        num_samples = min(comb(n, 6), 50000)
        total_sq_dev = 0.0
        for _ in range(num_samples):
            subset = rng.choice(n, size=6, replace=False)
            sub_adj = A[np.ix_(subset, subset)]
            degrees = sub_adj.sum(axis=1).astype(float)
            mean_deg = degrees.mean()
            total_sq_dev += np.sum((degrees - mean_deg) ** 2)
        
        return total_sq_dev / num_samples
>>>>>>> REPLACE

