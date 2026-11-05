"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    # 1. Run Greedy to get a good initial lower bound
    def run_greedy():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values, products

    best_seq, best_prods = run_greedy()
    best_len = len(best_seq)

    # Helper to compute all consecutive block products for a sequence
    def compute_products(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                prods.add(prod)
        return prods

    # Helper to check if adding x to seq creates a collision
    def can_append(seq, prods, x):
        tails = [x]
        prod = x
        for a in reversed(seq):
            prod *= a
            tails.append(prod)
        for t in tails:
            if t in prods:
                return False
        return True

    # 2. Build conflict graph: nodes are integers 2-128
    # Edge between a and b if they cannot both appear in the sequence
    # (i.e., some consecutive block product collision)
    # This is complex to precompute exactly, so we use a simpler approach:
    # Two numbers a < b conflict if gcd(a,b) > 1 in a way that creates collisions
    # Actually, the conflict depends on position, so we'll use a different strategy.

    # 3. Branch and bound with greedy coloring bound
    # We'll use a recursive search that tries to include or exclude each number
    # with a bound based on how many more numbers we can possibly add

    # Precompute: for each pair (a, b) with a < b, check if they can coexist
    # in some sequence. This is still complex, so let's use a heuristic.

    # Alternative: Use the beam search but with a better pruning strategy
    # and add a local search polish at the end.

    # 4. Beam Search with improved heuristics
    def check_validity(seq, prods, x):
        tails = [x]
        prod = x
        for a in reversed(seq):
            prod *= a
            tails.append(prod)
        for t in tails:
            if t in prods:
                return False, None
        new_prods = set(prods)
        new_prods.update(tails)
        return True, new_prods

    beam = {(): set()}
    W = 500  # Slightly larger beam width

    for x in range(2, 129):
        if time.time() > deadline - 5.0:  # Leave 5s for local search
            break

        next_beam = {}

        for seq, prods in beam.items():
            # Option 1: Skip x
            if seq not in next_beam:
                next_beam[seq] = prods

            # Option 2: Include x
            is_valid, new_prods = check_validity(seq, prods, x)

            if is_valid:
                new_seq = seq + (x,)
                next_beam[new_seq] = new_prods

        # Prune beam with better heuristic
        # Sort by: -len, then -sum (prefer larger sums for flexibility?), 
        # or use a score based on how "spread out" the products are
        def seq_score(s):
            if not s:
                return (0, 0)
            # Primary: length
            # Secondary: prefer sequences where the last element is smaller
            # (leaves room for more future elements)
            # Tertiary: sum
            return (-len(s), s[-1], sum(s))

        sorted_seqs = sorted(next_beam.keys(), key=seq_score)

        # Keep top W
        beam = {s: next_beam[s] for s in sorted_seqs[:W]}

        # Update best
        if sorted_seqs:
            top_seq = sorted_seqs[0]
            if len(top_seq) > best_len:
                best_len = len(top_seq)
                best_seq = list(top_seq)
                best_prods = next_beam[top_seq]

    # 5. Local search polish: try to improve the best sequence
    # Try removing one element and adding two others
    def try_improve(seq, prods):
        """Try to improve seq by removing one element and adding two others."""
        seq = list(seq)
        n = len(seq)
        improved = False

        # Try each removal position
        for i in range(n):
            if time.time() > deadline - 1.0:
                break
            # Remove seq[i]
            new_seq = seq[:i] + seq[i+1:]
            new_prods = compute_products(new_seq)
            
            # If new_seq is empty, min_val is 1, else last element
            min_val = new_seq[-1] if new_seq else 1

            # Try to add two elements from 2-128 not in new_seq
            existing = set(new_seq)
            # Ensure x is greater than the last element to maintain strict increase
            candidates = [x for x in range(max(2, min_val + 1), 129) if x not in existing]

            if not candidates:
                continue

            # Sort candidates by how "compatible" they are
            # (fewer product collisions with existing products)
            def candidate_score(x):
                tails = [x]
                prod = x
                for a in reversed(new_seq):
                    prod *= a
                    tails.append(prod)
                conflicts = sum(1 for t in tails if t in new_prods)
                return conflicts

            candidates.sort(key=candidate_score)

            # Try adding the top candidates
            for x in candidates[:20]:
                if time.time() > deadline - 1.0:
                    break
                if can_append(new_seq, new_prods, x):
                    seq2 = new_seq + [x]
                    prods2 = set(new_prods)
                    tails = [x]
                    prod = x
                    for a in reversed(new_seq):
                        prod *= a
                        tails.append(prod)
                    prods2.update(tails)

                    # Try adding one more
                    existing2 = set(seq2)
                    # y must be > x to maintain strict increase
                    candidates2 = [y for y in range(x + 1, 129) if y not in existing2]
                    for y in candidates2[:10]:
                        if can_append(seq2, prods2, y):
                            seq3 = seq2 + [y]
                            if len(seq3) > len(seq):
                                return seq3, compute_products(seq3), True

        return seq, prods, improved

    # Apply local search
    if time.time() < deadline - 2.0:
        best_seq, best_prods, _ = try_improve(best_seq, best_prods)
        best_len = len(best_seq)

    return list(best_seq)
# EVOLVE-BLOCK-END
