import math
import time
import random

def construct():
    deadline = time.time() + 110  # 10s safety margin
    best_score = 0.0
    best_solution = []
    
    moduli = list(range(20, 201))
    
    def calc_score(sol):
        return sum(1.0 / m for m, r in sol)
        
    def find_valid_residues(m, selected):
        forbidden = [False] * m
        for m_i, r_i in selected:
            g = math.gcd(m, m_i)
            target = r_i % g
            # Mark forbidden residues
            # Step size is g
            for r in range(target, m, g):
                forbidden[r] = True
        
        valid = []
        for r in range(m):
            if not forbidden[r]:
                valid.append(r)
        return valid

    def run_greedy(seed, strategy="random"):
        random.seed(seed)
        selected = []
        for m in moduli:
            valid_rs = find_valid_residues(m, selected)
            if valid_rs:
                if strategy == "random":
                    r = random.choice(valid_rs)
                elif strategy == "min":
                    r = valid_rs[0]
                elif strategy == "max":
                    r = valid_rs[-1]
                elif strategy == "mid":
                    r = valid_rs[len(valid_rs)//2]
                else:
                    r = random.choice(valid_rs)
                selected.append((m, r))
        return selected

    # Run deterministic strategies first
    for strat in ["min", "max", "mid"]:
        sol = run_greedy(0, strat)
        score = calc_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol

    # Run random restarts
    seed = 1
    while time.time() < deadline:
        sol = run_greedy(seed, "random")
        score = calc_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol
        seed += 1
        
        # If we have a very good score, we might break early?
        # Max possible is ~2.4. Score ~16.
        # If we get score > 10, it's very good.
        # But let's use the time.
        
    return [[m, r] for m, r in best_solution]
