<<<<<<< SEARCH
                    dist[(v, u)] = d + 1
                    queue.append((v, u))
                else:
                    # Found a cycle of length dist[(v,u)] + 1
                    # Wait, this isn't quite right for non-backtracking
                    # Let me reconsider
    
    # Simpler approach: BFS with parent tracking
    for start in range(n):
=======
                    dist[(v, u)] = d + 1
                    queue.append((v, u))
                else:
                    # Found a cycle of length dist[(v,u)] + 1
                    # Wait, this isn't quite right for non-backtracking
                    # Let me reconsider
                    pass

    # Simpler approach: BFS with parent tracking
    for start in range(n):
>>>>>>> REPLACE
