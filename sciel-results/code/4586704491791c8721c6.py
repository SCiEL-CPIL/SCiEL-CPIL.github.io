"""Maximize pairwise disjoint residue classes with distinct moduli 2..128.
Uses maximum clique branch-and-bound with coloring bound on the compatibility graph.
"""
import math
import time
import random

_START = None
MAX_MOD = 128


def _assign_residues(selected_moduli):
    """Assign residues for a compatible set of moduli."""
    n = len(selected_moduli)
    if n == 0:
        return []
    
    residues = [None] * n
    residues[0] = 0
    
    for i in range(1, n):
        m_i = selected_moduli[i]
        assigned = False
        for r in range(m_i):
            ok = True
            for j in range(i):
                m_j = selected_moduli[j]
                g = math.gcd(m_i, m_j)
                if g > 1 and (r - residues[j]) % g == 0:
                    ok = False
                    break
            if ok:
                residues[i] = r
                assigned = True
                break
        if not assigned:
            return None
    
    return list(zip(residues, selected_moduli))


def _assign_residues_backtrack(selected_moduli):
    """Backtracking assignment of residues."""
    n = len(selected_moduli)
    if n == 0:
        return []
    
    residues = [None] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        m = selected_moduli[idx]
        for r in range(m):
            ok = True
            for j in range(idx):
                g = math.gcd(m, selected_moduli[j])
                if g > 1 and (r - residues[j]) % g == 0:
                    ok = False
                    break
            if ok:
                residues[idx] = r
                if backtrack(idx + 1):
                    return True
        residues[idx] = None
        return False
    
    if backtrack(0):
        return list(zip(residues, selected_moduli))
    return None


def _greedy_max_clique(compat, n):
    """Greedy approximation for maximum clique."""
    best = set()
    candidates = set(range(n))
    
    while candidates:
        # Pick vertex with max degree in candidates
        v = max(candidates, key=lambda x: len(compat[x] & candidates))
        best.add(v)
        candidates = candidates & compat[v]
    
    return best


def construct():
    global _START
    if _START is None:
        _START = time.time()

    deadline = _START + 85.0

    moduli = list(range(2, MAX_MOD + 1))
    n_mod = len(moduli)
    
    # Build compatibility: two moduli are compatible if gcd > 1
    compat = [set() for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            if math.gcd(moduli[i], moduli[j]) > 1:
                compat[i].add(j)
                compat[j].add(i)
    
    # Greedy initial solution
    greedy_clique = _greedy_max_clique(compat, n_mod)
    best_clique = set(greedy_clique)
    best_size = len(greedy_clique)
    
    # Branch and bound with coloring
    deg = [len(compat[i]) for i in range(n_mod)]
    order = sorted(range(n_mod), key=lambda i: -deg[i])
    
    node_count = [0]
    
    def _greedy_color(P):
        """Assign colors to vertices in P. Returns dict v -> color (1-indexed)."""
        colors = {}
        P_list = sorted(P, key=lambda v: -len(compat[v] & P))
        for v in P_list:
            used_colors = set()
            for u in compat[v]:
                if u in colors:
                    used_colors.add(colors[u])
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
        return colors
    
    def _branch(P, current_clique):
        nonlocal best_clique, best_size, node_count
        
        node_count[0] += 1
        if time.time() > deadline - 2.0:
            return
        
        if not P:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = set(current_clique)
            return
        
        colors = _greedy_color(P)
        ub = max(colors.values()) if P else 0
        if len(current_clique) + ub <= best_size:
            return
        
        P_list = sorted(P, key=lambda v: -colors[v])
        
        for v in P_list:
            if len(current_clique) + colors[v] <= best_size:
                continue
            
            new_clique = current_clique + [v]
            new_P = P & compat[v]
            
            _branch(new_P, new_clique)
            
            P.discard(v)
    
    P = set(range(n_mod))
    _branch(P, [])
    
    selected_moduli = [moduli[i] for i in best_clique]
    
    # Assign residues
    pairs = _assign_residues(selected_moduli)
    if pairs is None:
        pairs = _assign_residues_backtrack(selected_moduli)
    if pairs is None:
        pairs = [(1, 2)]
    
    seen = set()
    out = []
    for r, m in pairs:
        if m in seen:
            continue
        if not (0 <= r < m):
            continue
        seen.add(m)
        out.append([r, m])
    
    if not out:
        out = [[1, 2]]
    
    return out
