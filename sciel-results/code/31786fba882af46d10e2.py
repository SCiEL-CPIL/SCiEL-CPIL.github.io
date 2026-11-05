"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    
    def has_3ap(s):
        """Check if set s contains a 3-term arithmetic progression."""
        lst = list(s)
        for i in range(len(lst)):
            for j in range(len(lst)):
                if i == j:
                    continue
                k_val = 2 * lst[j] - lst[i]
                if k_val in s and k_val != lst[i] and k_val != lst[j]:
                    return True
        return False
    
    def greedy_3ap_free(n):
        """Greedy construction of a 3-AP-free set in [0, n)."""
        points = set()
        for x in range(n):
            trial = points | {x}
            if not has_3ap(trial):
                points = trial
        return sorted(points)
    
    best_points = greedy_3ap_free(N)
    
    # Local search: try to add points by removing one and adding more
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = set(best_points)
        for p in list(current):
            trial = current - {p}
            for q in range(N):
                if q in trial:
                    continue
                if time.time() > deadline:
                    break
                new_trial = trial | {q}
                if not has_3ap(new_trial):
                    trial = new_trial
            if len(trial) > len(current):
                best_points = sorted(trial)
                improved = True
    
    return [(x, 0) for x in best_points]
# EVOLVE-BLOCK-END
