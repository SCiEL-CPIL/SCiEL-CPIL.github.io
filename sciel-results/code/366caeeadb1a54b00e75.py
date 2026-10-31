<<<<<<< SEARCH
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(p[i] - p[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Large multiplier ensures overlap is heavily penalized
        return max_S + 1000.0 * penalty
=======
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(p[i] - p[j])
                if d < SAFE_MIN_DIST:
                    penalty += (SAFE_MIN_DIST - d) ** 2
        
        # Very large multiplier ensures overlap is heavily penalized
        return max_S + 10000.0 * penalty
>>>>>>> REPLACE
