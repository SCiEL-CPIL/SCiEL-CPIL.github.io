      def can_add(u, v, adj_mask):
          # Check triples (u, a, b) where a, b in N(v)
          # v becomes a common neighbor of (u, a, b)
          # So we need |N(u) cap N(a) cap N(b)| <= 1
          for a in range(n):
              if not (adj_mask[v] & bit(a)): continue
              for b in range(a + 1, n):
                  if not (adj_mask[v] & bit(b)): continue
                  common = adj_mask[u] & adj_mask[a] & adj_mask[b]
                  if popcount(common) > 1:
                      return False
          
          # Check triples (v, a, b) where a, b in N(u)
          # u becomes a common neighbor of (v, a, b)
          # So we need |N(v) cap N(a) cap N(b)| <= 1
          for a in range(n):
              if not (adj_mask[u] & bit(a)): continue
              for b in range(a + 1, n):
                  if not (adj_mask[u] & bit(b)): continue
                  common = adj_mask[v] & adj_mask[a] & adj_mask[b]
                  if popcount(common) > 1:
                      return False
          
          return True
      