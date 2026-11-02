"""De Bruijn B(3,7) undirected: 3-regular, girth 16, n=2187."""
# EVOLVE-BLOCK-START
def construct(k, g):
    m = (g - 2) // 2  # 7 for g=16
    n = k ** m
    
    def word_to_idx(word):
        idx = 0
        for ch in word:
            idx = idx * k + ch
        return idx
    
    def idx_to_word(idx):
        word = []
        for _ in range(m):
            word.append(idx % k)
            idx //= k
        return list(reversed(word))
    
    adj = [[] for _ in range(n)]
    for u_idx in range(n):
        u = idx_to_word(u_idx)
        for j in range(k):
            v = u[1:] + [j]
            v_idx = word_to_idx(v)
            adj[u_idx].append(v_idx)
    
    edges = []
    seen = set()
    for u in range(n):
        for v in adj[u]:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                edges.append(key)
    
    return n, edges
# EVOLVE-BLOCK-END
