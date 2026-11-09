"""Rewritten program: DP with Pareto-optimal (length, last_gap) states for global optimality."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Sieve of Eratosthenes for primes <= 10000
    limit = 10000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = 0

    primes = np.array([p for p in range(2, limit + 1) if is_prime[p]], dtype=np.int32)
    n = len(primes)

    if n == 0:
        return []

    # For each prime i, maintain a list of Pareto-optimal states:
    # Each state is (length, last_gap, prev_state_id)
    # A state (L1, G1) dominates (L2, G2) if L1 >= L2 and G1 <= G2
    # We keep only non-dominated states.
    # states[i] = list of state IDs
    # all_states[sid] = (length, last_gap, prime_index, prev_state_id)
    
    all_states = []
    states_at = [[] for _ in range(n)]
    
    # Base case: prime[0] alone
    all_states.append((1, 0, 0, -1))
    states_at[0].append(0)
    
    best_len = 1
    best_sid = 0

    # For each prime i, find best predecessors j < i
    for i in range(1, n):
        if time.time() > deadline:
            break

        pi = int(primes[i])
        new_sids = []

        # Base: single prime i
        sid = len(all_states)
        all_states.append((1, 0, i, -1))
        new_sids.append(sid)

        # Try extending from each predecessor j < i
        for j in range(i):
            if time.time() > deadline:
                break
            gap = pi - int(primes[j])
            for sid_j in states_at[j]:
                L_j, G_j, _, _ = all_states[sid_j]
                if gap >= G_j:
                    new_sid = len(all_states)
                    all_states.append((L_j + 1, gap, i, sid_j))
                    new_sids.append(new_sid)
                    if L_j + 1 > best_len:
                        best_len = L_j + 1
                        best_sid = new_sid

        # Prune: keep only Pareto-optimal states at prime i
        # Sort by length desc, then gap asc
        new_sids.sort(key=lambda s: (-all_states[s][0], all_states[s][1]))
        pruned = []
        for sid in new_sids:
            L, G = all_states[sid][0], all_states[sid][1]
            dominated = False
            for sid_k in pruned:
                Lk, Gk = all_states[sid_k][0], all_states[sid_k][1]
                if Lk >= L and Gk <= G:
                    dominated = True
                    break
            if not dominated:
                pruned.append(sid)
        states_at[i] = pruned

    # Reconstruct using state IDs
    if best_sid < 0:
        return [int(primes[0])]

    seq = []
    cur_sid = best_sid
    while cur_sid >= 0:
        L, G, idx, prev_sid = all_states[cur_sid]
        seq.append(int(primes[idx]))
        cur_sid = prev_sid

    seq.reverse()
    return seq
# EVOLVE-BLOCK-END
