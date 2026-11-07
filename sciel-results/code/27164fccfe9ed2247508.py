"""Black-box optimizer for BBOB-style Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: lattice SA + (1+1)-ES with adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        calls = 0

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            return fx

        # Phase 1: Lattice-based search to exploit Rastrigin periodicity
        # Rastrigin has local minima at integer coordinates (after shift/rotation)
        # We systematically explore integer and half-integer lattice points
        best_x = None
        best_f = float('inf')

        # Generate lattice values: integers and half-integers in [-4, 4]
        lattice_values = []
        for i in range(-4, 5):
            lattice_values.append(float(i))
            lattice_values.append(float(i) + 0.5)

        # Build a structured set of candidates
        candidates = []

        # Center
        candidates.append(np.zeros(self.dim))

        # Pure integer points along each axis
        for d in range(self.dim):
            for val in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
                pt = np.zeros(self.dim)
                pt[d] = val
                candidates.append(pt)

        # Half-integer points along each axis
        for d in range(self.dim):
            for val in [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]:
                pt = np.zeros(self.dim)
                pt[d] = val
                candidates.append(pt)

        # Random lattice points (sample from the full lattice)
        n_random = 40
        for _ in range(n_random):
            pt = np.array([rng.choice(lattice_values) for _ in range(self.dim)])
            candidates.append(pt)

        # Deduplicate while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = tuple(np.round(c, 10))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        # Evaluate candidates
        for c in unique_candidates:
            if not check_time():
                break
            if calls >= self.budget:
                break
            fc = evaluate(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = evaluate(best_x)

        # Phase 2: Simulated Annealing on the lattice to jump between basins
        # This exploits the periodic structure: moving by 1 unit in any direction
        # shifts to an adjacent basin
        x = best_x.copy()
        fx = best_f
        T = 2.0  # Initial temperature
        cooling = 0.95
        sa_iterations = 150

        for it in range(sa_iterations):
            if calls >= self.budget or not check_time():
                break

            # Generate a lattice neighbor: shift one coordinate by ±1
            d = rng.integers(self.dim)
            direction = rng.choice([-1.0, 1.0])
            y = x.copy()
            y[d] += direction

            # If we're at a half-integer, snap to nearest integer for lattice SA
            # Actually, let's allow both integer and half-integer moves
            fy = evaluate(y)

            # Metropolis acceptance
            delta = fy - fx
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-8)):
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            T *= cooling

        # Phase 3: (1+1)-ES with adaptive step size for fine refinement
        x = best_x.copy()
        fx = best_f
        sigma = 0.3  # Start small since we're likely near a local min
        success_count = 0
        total_trials = 0
        window = 15

        while calls < self.budget and check_time():
            # Adaptive sigma: 1/5 rule
            if total_trials >= window:
                success_rate = success_count / total_trials
                if success_rate > 0.25:
                    sigma *= 1.2
                elif success_rate < 0.15:
                    sigma *= 0.8
                success_count = 0
                total_trials = 0

            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 0.005, 2.0)

            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5.0, 5.0)
            fy = evaluate(y)
            total_trials += 1

            if fy < fx:
                x = y
                fx = fy
                success_count += 1
                best_x = x.copy()
                best_f = fx
            else:
                # Periodic lattice jumps to escape local minima
                if total_trials % 25 == 0:
                    # Try a lattice point: round current position and shift
                    jump_base = np.round(x).astype(int).astype(float)
                    # Try shifting one coordinate by ±0.5 or ±1
                    d = rng.integers(self.dim)
                    offset = rng.choice([-1.0, -0.5, 0.5, 1.0])
                    jump = jump_base.copy()
                    jump[d] += offset
                    jump = np.clip(jump, -5.0, 5.0)
                    fj = evaluate(jump)
                    if fj < fx:
                        x = jump
                        fx = fj
                        best_x = x.copy()
                        best_f = fx
                        sigma = 0.5

        # Phase 4: Fine polishing with decreasing step sizes
        if calls < self.budget and check_time():
            sigma_fine = 0.05
            for _ in range(50):
                if calls >= self.budget or not check_time():
                    break
                y = x + sigma_fine * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy < fx:
                    x = y
                    fx = fy
                    best_x = x.copy()
                    best_f = fx
                sigma_fine *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
