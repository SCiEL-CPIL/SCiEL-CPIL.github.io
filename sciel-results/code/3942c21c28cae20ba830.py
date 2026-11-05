import time
import random
import math

N = 5000
START_TIME = time.time()
TIME_LIMIT = 325.0  # Safety margin before 330s kill

def _primes(limit):
    """Sieve of Eratosthenes to find primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def _greedy_construct(order, primes, start_time, time_limit):
    """Greedy construction given an ordering of candidates."""
    current_set = set()
    residue_counts = {}
    missing_residues = {}
    
    for x in order:
        if time.time() - start_time > time_limit - 5:
            break
        
        new_size = len(current_set) + 1
        blocked = False
        for p in primes:
            if p > new_size:
                break
            if p not in residue_counts:
                cnt = [0] * p
                for v in current_set:
                    cnt[v % p] += 1
                residue_counts[p] = cnt
                missing_residues[p] = set(i for i in range(p) if cnt[i] == 0)
            
            if len(missing_residues[p]) == 1:
                if x % p in missing_residues[p]:
                    blocked = True
                    break
        
        if not blocked:
            current_set.add(x)
            for p in primes:
                if p > len(current_set):
                    break
                if p not in residue_counts:
                    cnt = [0] * p
                    for v in current_set:
                        cnt[v % p] += 1
                    residue_counts[p] = cnt
                    missing_residues[p] = set(i for i in range(p) if cnt[i] == 0)
                else:
                    r = x % p
                    residue_counts[p][r] += 1
                    if residue_counts[p][r] == 1:
                        missing_residues[p].discard(r)
    
    return current_set

def _local_search(current_set, primes, start_time, time_limit):
    """Local search to improve the set."""
    current_set = set(current_set)
    not_in_set = list(set(range(N)) - current_set)
    random.shuffle(not_in_set)
    
    best_size = len(current_set)
    
    for pass_num in range(30):
        if time.time() - start_time > time_limit - 3:
            break
        
        k = len(current_set)
        
        # Rebuild residue info
        residue_counts = {}
        missing_residues = {}
        for p in primes:
            if p > k + 1:
                break
            cnt = [0] * p
            for v in current_set:
                cnt[v % p] += 1
            residue_counts[p] = cnt
            missing_residues[p] = set(i for i in range(p) if cnt[i] == 0)
        
        # Try to add
        added_any = False
        new_not_in_set = []
        for x in not_in_set:
            if time.time() - start_time > time_limit - 2:
                break
            if x in current_set:
                continue
            
            new_size = len(current_set) + 1
            blocked = False
            for p in primes:
                if p > new_size:
                    break
                if p not in residue_counts:
                    cnt = [0] * p
                    for v in current_set:
                        cnt[v % p] += 1
                    residue_counts[p] = cnt
                    missing_residues[p] = set(i for i in range(p) if cnt[i] == 0)
                
                if len(missing_residues[p]) == 1:
                    if x % p in missing_residues[p]:
                        blocked = True
                        break
            
            if not blocked:
                current_set.add(x)
                added_any = True
                for p in primes:
                    if p > len(current_set):
                        break
                    if p not in residue_counts:
                        cnt = [0] * p
                        for v in current_set:
                            cnt[v % p] += 1
                        residue_counts[p] = cnt
                        missing_residues[p] = set(i for i in range(p) if cnt[i] == 0)
                    else:
                        r = x % p
                        residue_counts[p][r] += 1
                        if residue_counts[p][r] == 1:
                            missing_residues[p].discard(r)
            else:
                new_not_in_set.append(x)
        
        not_in_set = new_not_in_set
        
        if not added_any:
            # Try swap: remove one, add two
            if len(current_set) > 0:
                swap_improved = False
                for _ in range(50):  # Try a few random swaps
                    if time.time() - start_time > time_limit - 2:
                        break
                    
                    # Pick a random element to remove
                    to_remove = random.choice(list(current_set))
                    current_set.discard(to_remove)
                    not_in_set.append(to_remove)
                    
                    # Try to add two elements
                    added_count = 0
                    k2 = len(current_set)
                    
                    # Rebuild
                    residue_counts2 = {}
                    missing_residues2 = {}
                    for p in primes:
                        if p > k2 + 2:
                            break
                        cnt = [0] * p
                        for v in current_set:
                            cnt[v % p] += 1
                        residue_counts2[p] = cnt
                        missing_residues2[p] = set(i for i in range(p) if cnt[i] == 0)
                    
                    shuffled = not_in_set[:]
                    random.shuffle(shuffled)
                    for x in shuffled[:200]:
                        if x in current_set:
                            continue
                        new_size = len(current_set) + 1
                        blocked = False
                        for p in primes:
                            if p > new_size:
                                break
                            if p not in residue_counts2:
                                cnt = [0] * p
                                for v in current_set:
                                    cnt[v % p] += 1
                                residue_counts2[p] = cnt
                                missing_residues2[p] = set(i for i in range(p) if cnt[i] == 0)
                            
                            if len(missing_residues2[p]) == 1:
                                if x % p in missing_residues2[p]:
                                    blocked = True
                                    break
                        
                        if not blocked:
                            current_set.add(x)
                            added_count += 1
                            for p in primes:
                                if p > len(current_set):
                                    break
                                if p not in residue_counts2:
                                    cnt = [0] * p
                                    for v in current_set:
                                        cnt[v % p] += 1
                                    residue_counts2[p] = cnt
                                    missing_residues2[p] = set(i for i in range(p) if cnt[i] == 0)
                                else:
                                    r = x % p
                                    residue_counts2[p][r] += 1
                                    if residue_counts2[p][r] == 1:
                                        missing_residues2[p].discard(r)
                        
                        if added_count >= 2:
                            break
                    
                    if added_count >= 2:
                        swap_improved = True
                        # Update not_in_set
                        not_in_set = [x for x in not_in_set if x not in current_set]
                        break
                    else:
                        # Revert
                        current_set.add(to_remove)
                        not_in_set.remove(to_remove)
                
                if not swap_improved:
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
    
    return current_set

def construct():
    global START_TIME
    start_time = START_TIME
    
    primes = _primes(N)
    
    best_set = set()
    best_size = 0
    
    # Multiple restarts with different orderings
    num_restarts = 40
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        # Create ordering
        if restart == 0:
            order = list(range(N))
        elif restart == 1:
            order = list(range(N))
            random.shuffle(order)
        else:
            # Mix of random and structured
            order = list(range(N))
            # Partial shuffle
            for i in range(N):
                j = random.randint(i, N - 1)
                order[i], order[j] = order[j], order[i]
        
        # Greedy phase
        current_set = _greedy_construct(order, primes, start_time, TIME_LIMIT)
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
        
        # Local search
        if time.time() - start_time < TIME_LIMIT - 5:
            current_set = _local_search(current_set, primes, start_time, TIME_LIMIT)
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
    
    return list(best_set)

# Initial call to set start time and warm up
construct()
