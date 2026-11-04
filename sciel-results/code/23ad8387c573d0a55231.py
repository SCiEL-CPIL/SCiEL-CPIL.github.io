"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    # Initialize with random +/- 1
    rng = np.random.RandomState(12345)
    h = rng.choice([-1, 1], size=N).astype(np.float64)

    def get_cost(h):
        # Compute aperiodic autocorrelation
        c = np.correlate(h, h, mode='full')
        return np.sum(c**2)

    current_cost = get_cost(h)
    best_cost = current_cost
    best_h = h.copy()

    # Phase 1: Local search with simulated annealing for better exploration
    temperature = 10000.0
    cooling_rate = 0.999
    min_temperature = 1.0

    while time.time() < deadline - 30.0:  # Leave 30s for final phase
        # Pick a random index to flip
        i = rng.randint(N)
        old_val = h[i]
        h[i] = -old_val

        new_cost = get_cost(h)
        delta = new_cost - current_cost

        if delta < 0:
            # Accept improvement
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
        else:
            # Accept worse move with probability exp(-delta/T)
            if temperature > min_temperature:
                prob = np.exp(-delta / temperature)
                if rng.random_sample() < prob:
                    current_cost = new_cost
                else:
                    h[i] = old_val
            else:
                h[i] = old_val

        # Cool down
        temperature *= cooling_rate

    # Phase 2: Multi-start local search from different random initializations
    remaining_time = deadline - time.time()
    num_restarts = min(50, int(remaining_time / 2.0))  # Estimate how many restarts we can afford

    for _ in range(num_restarts):
        if time.time() >= deadline:
            break

        # New random initialization
        h_restart = rng.choice([-1, 1], size=N).astype(np.float64)
        current_cost = get_cost(h_restart)

        # Quick local search on this restart
        restart_deadline = min(deadline, time.time() + max(1.0, remaining_time / (num_restarts + 1)))
        while time.time() < restart_deadline:
            i = rng.randint(N)
            old_val = h_restart[i]
            h_restart[i] = -old_val

            new_cost = get_cost(h_restart)

            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_h = h_restart.copy()
            else:
                h_restart[i] = old_val

    return best_h
# EVOLVE-BLOCK-END
