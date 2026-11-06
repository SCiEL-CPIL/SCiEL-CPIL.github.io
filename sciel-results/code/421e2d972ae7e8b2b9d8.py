    def simulate_greedy_fast(scores):
        selected = 0
        available = np.ones(n, dtype=bool)
        order = np.argsort(-scores, kind='stable')
        
        for v in order:
            if not available[v]:
                continue
            selected += 1
            available[v] = False
            nbrs = neighbors[v]
            if len(nbrs) > 0:
                available[nbrs] = False
        
        return selected
    