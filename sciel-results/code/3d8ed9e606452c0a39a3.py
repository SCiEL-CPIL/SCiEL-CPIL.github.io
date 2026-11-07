"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of restarting (1+1)-ES agents with adaptive step sizes.

    Uses a mix of exploration and exploitation strategies, with periodic
    restarts based on stall detection. All evaluations share a single budget.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # safety margin vs 330 s wall-clock kill

        rng = np.random.default_rng(42)

        # ---- global best bookkeeping -------------------------------------
        x_best = None
        f_best = np.inf
        n_calls = 0

        def record(x, f):
            nonlocal x_best, f_best, n_calls
            if f < f_best:
                f_best = f
                x_best = x.copy()
            n_calls += 1
            return f_best

        # initial diverse seeds
        n_seeds = min(10, budget // 5)
        seeds = [rng.uniform(-5.0, 5.0, dim) for _ in range(n_seeds)]
        for s in seeds:
            f = func(s)
            record(s, f)
            if n_calls >= budget or time.time() > deadline:
                break

        if n_calls >= budget or time.time() > deadline:
            return x_best, f_best

        # ---- portfolio of (1+1)-ES agents --------------------------------
        # Mix of step sizes: large (exploration) to small (exploitation).
        sigmas = [3.0, 1.5, 0.75, 0.375, 0.1875, 0.09]
        
        agents = []
        for i, sg in enumerate(sigmas):
            if i < 2:
                # exploiters start near the current global best
                x = x_best + rng.standard_normal(dim) * 0.5
            elif i < 4:
                # mid-range start
                x = x_best + rng.standard_normal(dim) * 1.0
            else:
                # explorers start fresh
                x = rng.uniform(-5.0, 5.0, dim)
            x = np.clip(x, -5.0, 5.0)
            agents.append({
                "x": x,
                "sigma": sg,
                "best_local": np.inf,
                "stall": 0,
                "success_count": 0,
                "total_steps": 0,
                "restart_count": 0
            })

        cursor = 0
        def next_agent():
            nonlocal cursor
            a = agents[cursor % len(agents)]
            cursor += 1
            return a

        # Main search loop
        while n_calls < budget and time.time() < deadline:
            a = next_agent()
            
            # Adaptive sigma adjustment based on recent success rate
            if a["total_steps"] > 10:
                success_rate = a["success_count"] / a["total_steps"]
                # 1/5 rule: increase sigma if success rate > 0.2, decrease if < 0.2
                if success_rate > 0.3:
                    a["sigma"] = min(a["sigma"] * 1.1, 5.0)
                elif success_rate < 0.1:
                    a["sigma"] = max(a["sigma"] * 0.9, 0.01)
                
                # Reset counters periodically
                if a["total_steps"] % 20 == 0:
                    a["success_count"] = 0
                    a["total_steps"] = 0

            # Propose a mutant
            step = a["sigma"] * rng.standard_normal(dim)
            y = np.clip(a["x"] + step, -5.0, 5.0)
            fy = func(y)
            record(y, fy)
            
            a["total_steps"] += 1
            
            if fy < a["best_local"]:
                a["best_local"] = fy
                a["stall"] = 0
                a["success_count"] += 1
                # Accept, move the agent
                a["x"] = y
            else:
                a["stall"] += 1
                # Restart logic
                if a["stall"] >= 40:
                    a["restart_count"] += 1
                    if a["sigma"] > 1.0:
                        # Explorer: jump somewhere new but biased toward best
                        a["x"] = np.clip(x_best + rng.standard_normal(dim) * 1.5, -5.0, 5.0)
                    else:
                        # Exploiter: tighten around global best
                        a["x"] = np.clip(x_best + rng.standard_normal(dim) * 0.3, -5.0, 5.0)
                    a["stall"] = 0
                    a["sigma"] = min(a["sigma"] * 1.2, 4.0)
                    a["best_local"] = np.inf
                    a["success_count"] = 0
                    a["total_steps"] = 0

        return x_best, f_best
# EVOLVE-BLOCK-END
