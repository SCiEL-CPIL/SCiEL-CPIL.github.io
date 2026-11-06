   # dist_current: (N,)
   # dist_unvisited: (N, N) -> dist[unvisited, unvisited]
   # We need min of each row, excluding diagonal.
   # Set diagonal to inf, then take min along axis 1.
   