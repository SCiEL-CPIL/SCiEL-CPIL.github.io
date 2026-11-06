import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    all_moduli = list(range(2, 129))
    # Precompute gcds for speed
    gcds = {}
    for i in range(len(all_moduli)):
        for j in range(i + 1, len(all_moduli)):
            m1, m2 = all_moduli[i], all_moduli[j]
            gcds[(m1, m2)] = math.gcd(m1, m2)
            gcds[(m2, m1)] = gcds[(m1, m2)]

    def disjoint(r1, m1, r2, m2):
        """Check if residue classes x ≡ r1 (mod m1) and x ≡ r2 (mod m2) are disjoint."""
        g = gcds.get((m1, m2), math.gcd(m1, m2))
        return (r1 - r2) % g != 0

    def get_valid_residues(m, assigned):
        """
        Get all residues r for modulus m that are disjoint from all assigned classes.
        assigned is a list of (r, m) tuples.
        """
        valid = []
        for r in range(m):
            is_valid = True
            for r2, m2 in assigned:
                if not disjoint(r, m, r2, m2):
                    is_valid = False
                    break
            if is_valid:
                valid.append(r)
        return valid

    def forward_check(assigned, unassigned):
        """
        Check if any modulus in unassigned has 0 valid residues given 'assigned'.
        Returns (is_consistent, blocked_moduli)
        blocked_moduli is the set of moduli with 0 valid residues.
        """
        blocked = set()
        for m in unassigned:
            # Optimization: if we've already blocked this in a previous check? 
            # No, we recompute based on current assigned.
            valids = get_valid_residues(m, assigned)
            if not valids:
                blocked.add(m)
        return len(blocked) == 0, blocked

    best_solution = []
    best_count = 0
    search_calls = 0

    def search(assigned, unassigned):
        nonlocal best_solution, best_count, search_calls, start_time, deadline

        if time.time() > deadline - 0.5:
            return

        search_calls += 1
        if search_calls > 500000: # Hard cap on nodes to prevent infinite loop if time check fails
            return

        # Prune unassigned moduli that are already impossible?
        # Actually, we should check consistency first.
        # If a modulus has no valid residues, we can't add it.
        # We can either remove it from consideration (accepting we can't add it)
        # or backtrack.
        # In this formulation, we are trying to ADD moduli.
        # If a modulus in `unassigned` has 0 valid residues, it is "dead".
        # We can remove it from `unassigned` and continue, BUT this means we failed to include it.
        # Since we want to maximize count, we should try to avoid making them dead.
        # However, if they are dead, we just skip them.
        
        # Let's identify dead moduli
        dead = set()
        valid_counts = {}
        for m in unassigned:
            valids = get_valid_residues(m, assigned)
            if not valids:
                dead.add(m)
            else:
                valid_counts[m] = valids

        # If all remaining are dead, we are done with this branch
        if not valid_counts:
            if len(assigned) > best_count:
                best_count = len(assigned)
                best_solution = list(assigned)
            return

        # MRV Heuristic: Pick modulus with fewest valid residues
        # This forces a decision on the most constrained variable
        m_target = min(valid_counts.keys(), key=lambda x: len(valid_counts[x]))
        residues = valid_counts[m_target]
        
        # Shuffle residues to explore different branches
        random.shuffle(residues)

        # Branch on each valid residue
        for r in residues:
            new_assigned = assigned + [(r, m_target)]
            new_unassigned = unassigned - {m_target}
            
            # Forward Checking: 
            # Check if this choice makes any other modulus dead.
            # If it does, we can still proceed, but those moduli will be dropped.
            # To be more aggressive, we can check if the *number* of dead moduli increases significantly?
            # For now, just recurse. The next call will filter out the dead ones.
            
            search(new_assigned, new_unassigned)
            
            if time.time() > deadline - 0.5:
                return

    # Initial call
    # To improve performance, we can start with a small set of "anchor" moduli
    # or just start empty. Starting empty is safest for maximizing count.
    search([], set(all_moduli))

    # If the search didn't find anything (should not happen), fallback to simple greedy
    if not best_solution:
        assigned = []
        for m in all_moduli:
            valids = get_valid_residues(m, assigned)
            if valids:
                # Pick a random valid residue
                r = random.choice(valids)
                assigned.append((r, m))
        best_solution = assigned
        best_count = len(assigned)

    result = [[r, m] for r, m in best_solution]
    
    # Fallback if empty
    if not result:
        result = [[0, 2]]
        
    return result

# Quick sanity check for disjointness
def verify(result):
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            r1, m1 = result[i]
            r2, m2 = result[j]
            g = math.gcd(m1, m2)
            if (r1 - r2) % g == 0:
                return False
    return True

if __name__ == "__main__":
    res = construct()
    if not verify(res):
        print("Verification Failed")
    else:
        print(f"Valid solution with {len(res)} classes")
