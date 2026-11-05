"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    def is_compatible(assignment, m, r):
        for (cm, cr) in assignment:
            g = math.gcd(cm, m)
            if cr % g == r % g:
                return False
        return True
    
    # Greedy: add moduli in order of increasing m, pick first compatible residue
    best = []
    for m in range(20, 201):
        for r in range(m):
            if is_compatible(best, m, r):
                best.append((m, r))
                break
    
    score = sum(1.0/m for m, _ in best)
    
    # Local search
    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            bm, br = best[i]
            for m in range(20, 201):
                if any(cm == m for cm, _ in best):
                    continue
                for r in range(m):
                    new_best = best[:i] + best[i+1:]
                    if is_compatible(new_best, m, r):
                        new_score = score - 1.0/bm + 1.0/m
                        if new_score > score + 1e-9:
                            best = new_best + [(m, r)]
                            score = new_score
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return [[m, r] for m, r in best]

# EVOLVE-BLOCK-END
