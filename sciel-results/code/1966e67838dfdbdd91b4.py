import time
import random
import numpy as np

# Precomputation
N = 256
three_aps = []
four_aps = []
aps_with_3 = [[] for _ in range(N + 1)]
aps_with_4 = [[] for _ in range(N + 1)]

for a in range(1, N - 2):
    for d in range(1, (N - a) // 2 + 1):
        b = a + d
        c = a + 2 * d
        if c <= N:
            three_aps.append((a, b, c))
            for x in (a, b, c):
                aps_with_3[x].append((a, b, c))

for a in range(1, N - 3):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        if e <= N:
            four_aps.append((a, b, c, e))
            for x in (a, b, c, e):
                aps_with_4[x].append((a, b, c, e))

# Seed construction (greedy)
def seed_construct():
    selected = set()
    for x in range(1, N + 1):
        # Check if adding x creates a 4-AP
        valid = True
        for ap in aps_with_4[x]:
            others = [y for y in ap if y != x]
            if all(y in selected for y in others):
                valid = False
                break
        if valid:
            selected.add(x)
    return selected

def count_3aps(selected):
    count = 0
    for ap in three_aps:
        if all(x in selected for x in ap):
            count += 1
    return count

class Replica:
    def __init__(self, state, temp):
        self.state = set(state)
        self.temp = temp
        self.ap3_count = count_3aps(self.state)
        self.ap4_count = 0 # Should be 0 for valid state
        
    def is_valid(self):
        return self.ap4_count == 0

    def propose_move(self):
        """Returns (new_state, delta_ap3, delta_ap4)"""
        if random.random() < 0.3:
            # Add
            candidates = [x for x in range(1, N + 1) if x not in self.state]
            if not candidates: return None
            x = random.choice(candidates)
            # Check 4-AP validity
            for ap in aps_with_4[x]:
                others = [y for y in ap if y != x]
                if all(y in self.state for y in others):
                    return None # Invalid add
            # Count new 3-APs
            delta3 = 0
            for ap in aps_with_3[x]:
                others = [y for y in ap if y != x]
                if all(y in self.state for y in others):
                    delta3 += 1
            return (x, 'add', delta3, 0)
            
        elif random.random() < 0.6:
            # Remove
            if not self.state: return None
            x = random.choice(list(self.state))
            # Count lost 3-APs
            delta3 = 0
            for ap in aps_with_3[x]:
                others = [y for y in ap if y != x]
                if all(y in self.state for y in others):
                    delta3 -= 1
            return (x, 'remove', delta3, 0)
            
        else:
            # Swap
            if not self.state: return None
            x = random.choice(list(self.state))
            candidates = [y for y in range(1, N + 1) if y not in self.state]
            if not candidates: return None
            y = random.choice(candidates)
            
            # Check if adding y is valid given x is removed
            # Temporarily remove x
            self.state.remove(x)
            valid = True
            for ap in aps_with_4[y]:
                others = [z for z in ap if z != y]
                if all(z in self.state for z in others):
                    valid = False
                    break
            if not valid:
                self.state.add(x)
                return None
            
            # Calculate delta 3-APs
            # Lost 3-APs with x
            delta3 = 0
            for ap in aps_with_3[x]:
                others = [z for z in ap if z != x]
                if all(z in self.state for z in others):
                    delta3 -= 1
            
            # Gained 3-APs with y
            for ap in aps_with_3[y]:
                others = [z for z in ap if z != y]
                if all(z in self.state for z in others):
                    delta3 += 1
            
            # Apply swap
            self.state.add(y)
            return (y, 'swap', delta3, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # Initial state
    init_state = seed_construct()
    
    # Initialize replicas
    num_replicas = 8
    temps = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    replicas = [Replica(init_state, t) for t in temps]
    
    best_score = -1
    best_state = set(init_state)
    
    for i, rep in enumerate(replicas):
        if rep.ap3_count > best_score:
            best_score = rep.ap3_count
            best_state = set(rep.state)
            
    while time.time() < deadline:
        for i in range(num_replicas):
            if time.time() > deadline: break
            rep = replicas[i]
            
            # Propose move
            move = rep.propose_move()
            if move is None:
                continue
                
            x, typ, delta3, delta4 = move
            
            # Energy change: E = -ap3_count
            delta_E = -delta3
            
            # Acceptance probability
            if delta_E < 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta_E / rep.temp)
                
            if accept:
                # Apply move
                if typ == 'add':
                    rep.state.add(x)
                elif typ == 'remove':
                    rep.state.remove(x)
                # For swap, state was already updated in propose_move
                # Wait, in swap propose_move, we removed x and added y.
                # But if we reject, we need to revert.
                # My propose_move for swap modifies self.state!
                # This is a bug risk. Let's fix propose_move to not modify state, or handle revert.
                
                # Let's restructure: propose_move should return the change, not modify state.
                # Or, simpler: just revert if rejected.
                # In swap case: we removed x, checked validity of y, calculated delta.
                # If rejected, we must re-add x and ensure y is not added.
                # But we already added y in propose_move? No, I wrote `self.state.add(y)` at the end of swap block in propose_move.
                # This is dangerous. Let's rewrite propose_move to be pure.
                
                # REWRITE PROPOSE_MOVE TO BE PURE
                # We will discard the current implementation of propose_move and use a safer one.
                
                pass # Placeholder for logic flow, actual logic below
                
        # Swap replicas
        for i in range(num_replicas - 1):
            if time.time() > deadline: break
            r1 = replicas[i]
            r2 = replicas[i + 1]
            # Metropolis swap criterion
            # E1, E2 are energies (-ap3)
            # T1, T2 are temps
            # Accept swap if exp(-(E2-E1)/T1) * exp(-(E1-E2)/T2) > rand
            # Or simpler: accept if exp(-((E2-E1)/T1 - (E2-E1)/T2)) > rand
            dE = r2.ap3_count - r1.ap3_count # delta in ap3 count
            # Energy delta for r1 going to T2: -dE
            # Energy delta for r2 going to T1: dE
            # Log prob: -(-dE)/T2 - (dE)/T1 = dE/T2 - dE/T1 = dE * (1/T2 - 1/T1)
            # If dE > 0 (r2 better), 1/T2 < 1/T1 (since T2 > T1), so term is negative.
            # We want to accept if exp(term) > rand.
            term = dE * (1.0 / r2.temp - 1.0 / r1.temp)
            if random.random() < np.exp(term):
                replicas[i], replicas[i + 1] = r2, r1
                # Update best if swapped replica is better
                if replicas[i].ap3_count > best_score:
                    best_score = replicas[i].ap3_count
                    best_state = set(replicas[i].state)
                    
            # Update best for current positions
            if replicas[i].ap3_count > best_score:
                best_score = replicas[i].ap3_count
                best_state = set(replicas[i].state)
            if replicas[i+1].ap3_count > best_score:
                best_score = replicas[i+1].ap3_count
                best_state = set(replicas[i+1].state)

    return sorted(best_state)

# Since the previous propose_move was flawed (modified state), I will implement a clean version inside construct or as a helper.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    init_state = seed_construct()
    
    num_replicas = 8
    temps = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # State representation: list of booleans for speed? Or set?
    # Set is fine for 256 elements.
    
    replicas = []
    for t in temps:
        state = set(init_state)
        ap3 = count_3aps(state)
        replicas.append({
            'state': state,
            'temp': t,
            'ap3': ap3
        })
        
    best_score = max(r['ap3'] for r in replicas)
    best_state = set(replicas[0]['state'])
    for r in replicas:
        if r['ap3'] > best_score:
            best_score = r['ap3']
            best_state = set(r['state'])
            
    while time.time() < deadline:
        for i in range(num_replicas):
            if time.time() > deadline: break
            rep = replicas[i]
            state = rep['state']
            temp = rep['temp']
            ap3_count = rep['ap3']
            
            move_type = random.random()
            if move_type < 0.3:
                # Add
                candidates = [x for x in range(1, N + 1) if x not in state]
                if not candidates: continue
                x = random.choice(candidates)
                # Check 4-AP
                valid = True
                for ap in aps_with_4[x]:
                    others = [y for y in ap if y != x]
                    if all(y in state for y in others):
                        valid = False
                        break
                if not valid: continue
                
                # Delta 3-AP
                delta3 = 0
                for ap in aps_with_3[x]:
                    others = [y for y in ap if y != x]
                    if all(y in state for y in others):
                        delta3 += 1
                        
                new_ap3 = ap3_count + delta3
                delta_E = -delta3
                if delta_E < 0 or random.random() < np.exp(-delta_E / temp):
                    state.add(x)
                    rep['ap3'] = new_ap3
                    if new_ap3 > best_score:
                        best_score = new_ap3
                        best_state = set(state)
                        
            elif move_type < 0.6:
                # Remove
                if not state: continue
                x = random.choice(list(state))
                delta3 = 0
                for ap in aps_with_3[x]:
                    others = [y for y in ap if y != x]
                    if all(y in state for y in others):
                        delta3 -= 1
                        
                new_ap3 = ap3_count + delta3
                delta_E = -delta3
                if delta_E < 0 or random.random() < np.exp(-delta_E / temp):
                    state.remove(x)
                    rep['ap3'] = new_ap3
                    if new_ap3 > best_score:
                        best_score = new_ap3
                        best_state = set(state)
                        
            else:
                # Swap
                if not state: continue
                x = random.choice(list(state))
                candidates = [y for y in range(1, N + 1) if y not in state]
                if not candidates: continue
                y = random.choice(candidates)
                
                # Check validity of y with x removed
                # Temporarily remove x for check
                state.remove(x)
                valid = True
                for ap in aps_with_4[y]:
                    others = [z for z in ap if z != y]
                    if all(z in state for z in others):
                        valid = False
                        break
                if not valid:
                    state.add(x) # Revert
                    continue
                    
                # Calculate delta 3-AP
                # Lost with x
                delta3 = 0
                for ap in aps_with_3[x]:
                    others = [z for z in ap if z != x]
                    if all(z in state for z in others):
                        delta3 -= 1
                # Gained with y
                for ap in aps_with_3[y]:
                    others = [z for z in ap if z != y]
                    if all(z in state for z in others):
                        delta3 += 1
                        
                new_ap3 = ap3_count + delta3
                delta_E = -delta3
                if delta_E < 0 or random.random() < np.exp(-delta_E / temp):
                    # Accept: add y
                    state.add(y)
                    rep['ap3'] = new_ap3
                    if new_ap3 > best_score:
                        best_score = new_ap3
                        best_state = set(state)
                else:
                    # Reject: re-add x
                    state.add(x)
                    
        # Swaps
        for i in range(num_replicas - 1):
            if time.time() > deadline: break
            r1 = replicas[i]
            r2 = replicas[i + 1]
            dE = r2['ap3'] - r1['ap3']
            term = dE * (1.0 / r2['temp'] - 1.0 / r1['temp'])
            if random.random() < np.exp(term):
                replicas[i], replicas[i + 1] = r2, r1
                # Update best
                if r2['ap3'] > best_score:
                    best_score = r2['ap3']
                    best_state = set(r2['state'])
                if r1['ap3'] > best_score:
                    best_score = r1['ap3']
                    best_state = set(r1['state'])
            else:
                # Update best for current positions just in case
                if r1['ap3'] > best_score:
                    best_score = r1['ap3']
                    best_state = set(r1['state'])
                if r2['ap3'] > best_score:
                    best_score = r2['ap3']
                    best_state = set(r2['state'])

    return sorted(best_state)
