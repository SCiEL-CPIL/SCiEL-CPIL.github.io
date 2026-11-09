    # Refinement: Local search on the best solution found
    # We can try to reduce S further by fine-tuning positions
    # But the random search with cooling usually does this.
    
    # Let's ensure we return a valid solution.
    # If overlap is still present, we might need to expand the inner hexagons?
    # No, they are fixed size.
    # If overlap is present, the configuration is invalid.
    # We should try to fix it by pushing overlapping hexagons apart.
    
    # Simple fix:
    for _ in range(100):
        overlap = check_overlaps(best_pos)
        if overlap < 1e-6:
            break
        # Find a pair that overlaps and push them apart
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_pos[i] - best_pos[j])
                if d < MIN_DIST:
                    # Push apart
                    diff = best_pos[i] - best_pos[j]
                    if np.linalg.norm(diff) < 1e-6:
                        diff = np.array([1.0, 0.0])
                    else:
                        diff = diff / np.linalg.norm(diff)
                    push = (MIN_DIST - d) * 0.5 + 0.01
                    best_pos[i] += diff * push
                    best_pos[j] -= diff * push
