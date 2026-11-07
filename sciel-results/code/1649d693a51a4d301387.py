          while qi < len(queue):
              u = queue[qi]; qi += 1
              du = dist[u]
              if du >= 4:
                  continue
              for v in adj[u]:
                  if dist[v] == -1:
                      dist[v] = du + 1
                      if dist[v] > 4:
                          return False
                      queue.append(v)
  