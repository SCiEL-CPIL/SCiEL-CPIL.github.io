import time
import random
from collections import defaultdict
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    N = 192
    K = 3
    MAX_SIZE = 48
    
    # Precompute: for each a in 1..192, which triples (b,c,d) of distinct values in 1..192
    # (with b != a, c != a, d != a, all distinct) sum to a multiple of a.
    # We'll represent conflicts as: for a given a, the set of triples that are "bad".
    # But that's too many. Instead, we'll use a different approach.
    
    # For each a, precompute all pairs (b, c) with b < c, b != a, c != a, such that
    # there exists d != a, d != b, d != c with (b + c + d) % a == 0.
    # Actually, we need: for a given set S containing a, no triple from S\{a} sums to multiple of a.
    
    # Better approach: for each a, precompute for each pair (b, c) in 1..192 (b != a, c != a, b != c),
    # whether there exists a d in 1..192 (d != a, d != b, d != c) such that (b + c + d) % a == 0.
    # If yes, then if b, c, d are all in S, the constraint is violated for a.
    
    # This is still complex. Let's use a direct check function that's optimized.
    
    def can_add_to_set(S, x):
        """Check if x can be added to set S without violating constraints."""
        S_list = list(S)
        n = len(S_list)
        
        # Check 1: For a = x, no triple in S sums to a multiple of x
        if n >= 3:
            # For each pair (b, c) in S, check if there exists d in S (d != b, d != c)
            # such that (b + c + d) % x == 0
            residue_count = defaultdict(int)
            for v in S_list:
                residue_count[v % x] += 1
            
            for i in range(n):
                b = S_list[i]
                for j in range(i + 1, n):
                    c = S_list[j]
                    need = (-b - c) % x
                    cnt = residue_count.get(need, 0)
                    # Subtract b and c if they have the needed residue
                    avail = cnt
                    if b % x == need:
                        avail -= 1
                    if c % x == need:
                        avail -= 1
                    if avail > 0:
                        return False
        
        # Check 2: For each a in S, no pair (b, c) in S\{a} with (b + c + x) % a == 0
        for a in S_list:
            if a == x:
                continue
            # Check if there exists b, c in S\{a} (b != c) such that (b + c + x) % a == 0
            # i.e., (b + c) % a == (-x) % a
            target = (-x) % a
            # Count residues of elements in S\{a}
            residue_count = defaultdict(int)
            for v in S_list:
                if v != a:
                    residue_count[v % a] += 1
            
            for i in range(n):
                b = S_list[i]
                if b == a:
                    continue
                for j in range(i + 1, n):
                    c = S_list[j]
                    if c == a:
                        continue
                    if (b + c) % a == target:
                        return False
        
        return True
    
    def is_valid_set(S):
        """Verify that S is a valid set."""
        S_list = list(S)
        n = len(S_list)
        for a in S_list:
            others = [x for x in S_list if x != a]
            m = len(others)
            if m < 3:
                continue
            residue_count = defaultdict(int)
            for x in others:
                residue_count[x % a] += 1
            for i in range(m):
                b = others[i]
                for j in range(i + 1, m):
                    c = others[j]
                    need = (-b - c) % a
                    cnt = residue_count.get(need, 0)
                    avail = cnt
                    if b % a == need:
                        avail -= 1
                    if c % a == need:
                        avail -= 1
                    if avail > 0:
                        return False
        return True
    
    # Strategy: Use a smarter greedy with precomputed conflict info
    # Key insight: larger numbers have fewer multiples, so they're "safer"
    # Also, numbers that are prime or have large prime factors are safer
    
    # Precompute for each x in 1..192, how "conflicting" it is
    # A number x conflicts with a set S if there exist b, c, d in S (all != x, distinct)
    # such that (b + c + d) % x == 0, OR for some a in S, there exist b, c in S\{a} 
    # such that (b + c + x) % a == 0.
    
    # Let's try multiple strategies:
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Greedy from large to small
    def greedy_build(order, max_size=MAX_SIZE, time_limit=None):
        selected = []
        sel_set = set()
        for x in order:
            if len(selected) >= max_size:
                break
            if time_limit is not None and time.time() > time_limit:
                break
            if can_add_to_set(sel_set, x):
                selected.append(x)
                sel_set.add(x)
        return selected
    
    # Strategy 2: Greedy from small to large
    # Strategy 3: Greedy from "safe" numbers (large primes, etc.)
    
    random.seed(42)
    
    # Try various orders
    orders = [
        list(range(N, 0, -1)),  # Large to small
        list(range(1, N + 1)),  # Small to large
        [x for x in range(N, 0, -1) if x > N // 2],  # Top half only
        [x for x in range(N, 0, -1) if x > N // 3],  # Top two-thirds
    ]
    
    # Add random orders
    for _ in range(20):
        order = list(range(1, N + 1))
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() > deadline:
            break
        s = greedy_build(order)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s
    
    # Local search: try to add more elements
    current_set = set(best_set)
    full_range = set(range(1, N + 1))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        complement = full_range - current_set
        
        # Try single additions
        for y in sorted(complement, reverse=True):
            if time.time() > deadline:
                break
            if len(current_set) >= MAX_SIZE:
                break
            if can_add_to_set(current_set, y):
                current_set.add(y)
                best_set = list(current_set)
                best_len = len(current_set)
                improved = True
                break
        
        if improved:
            continue
        
        # Try remove-1-add-2 moves
        comp_list = list(complement)
        random.shuffle(comp_list)
        
        found = False
        for x in list(current_set):
            if time.time() > deadline:
                break
            S_minus_x = current_set - {x}
            
            for y1 in comp_list[:30]:
                if not can_add_to_set(S_minus_x, y1):
                    continue
                S_temp = S_minus_x | {y1}
                for y2 in comp_list[:30]:
                    if y1 == y2:
                        continue
                    if can_add_to_set(S_temp, y2):
                        new_set = S_temp | {y2}
                        # Try to extend
                        comp2 = full_range - new_set
                        for y3 in sorted(comp2, reverse=True):
                            if time.time() > deadline:
                                break
                            if len(new_set) >= MAX_SIZE:
                                break
                            if can_add_to_set(new_set, y3):
                                new_set.add(y3)
                                break
                        if len(new_set) > best_len:
                            current_set = new_set
                            best_set = list(current_set)
                            best_len = len(current_set)
                            improved = True
                            found = True
                            break
                if found:
                    break
            if found:
                break
    
    # Final verification
    if not is_valid_set(set(best_set)):
        # Fallback
        best_set = greedy_build(list(range(N, 0, -1)))
    
    return best_set
