      def _min_triangle_area_fast(points):
          n = len(points)
          # Create indices for all triples
          # This might be memory heavy for large N, but N=13 is small.
          # Number of triples: 286.
          # We can just loop over i, j, k in Python? 286 iterations is fast.
          # But we can do it in one go?
          # Let's stick to a tight Python loop or small vectorized ops.
          min_area = np.inf
          for i in range(n):
              for j in range(i+1, n):
                  for k in range(j+1, n):
                      v1 = points[j] - points[i]
                      v2 = points[k] - points[i]
                      cross = v1[0]*v2[1] - v1[1]*v2[0]
                      area = 0.5 * abs(cross)
                      if area < min_area:
                          min_area = area
          return min_area
      