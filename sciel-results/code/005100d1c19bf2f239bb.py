"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    TARGET = 200
    ALL = list(range(1, 101))
    MASK_LIMIT = (1 << TARGET) - 1 # Mask to keep bitset within 0..TARGET-1? 
    # Actually, we only care about sums <= TARGET.
    # If a sum exceeds TARGET, it doesn't help form TARGET (since all nums positive).
    # So we can mask with (1 << TARGET) - 1 to keep ints small?
    # 1 << 200 is a 201-bit integer. Python handles this fine.
    
    def greedy(order):
        """
        Greedy construction using bitmask for reachable sums.
        """
        S = []
        reachable = 1  # Bit 0 is set (sum 0 is reachable)
        
        for x in order:
            # Check if adding x creates sum TARGET
            # We need to check if (TARGET - x) is currently reachable
            # i.e., bit (TARGET - x) in reachable is 1
            # If TARGET - x < 0, it's impossible, so safe.
            if x < TARGET:
                if (reachable >> (TARGET - x)) & 1:
                    continue
            
            # Add x
            # New reachable sums are old sums and old sums + x
            reachable = reachable | (reachable << x)
            # Optional: Mask to prevent integer growth? 
            # Sums > TARGET don't contribute to sum TARGET in future steps (positive nums).
            # So we can mask with (1 << TARGET) - 1.
            # But (1 << 200) is small. Let's keep it simple.
            
            S.append(x)
            
        return S

    def local_improve_add_only(S):
        """
        Try to add back excluded elements to S if possible using bitmask.
        """
        S_set = set(S)
        reachable = 1
        for x in S:
            reachable = reachable | (reachable << x)
            
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try to add any missing element
            # Sort missing by descending value? Large values are safer.
            missing = [x for x in ALL if x not in S_set]
            # Randomize missing order to avoid bias? Or sort?
            # Sorting by descending value is a good heuristic for "safety".
            for x in sorted(missing, reverse=True):
                if x < TARGET:
                    if (reachable >> (TARGET - x)) & 1:
                        continue
                # Can add x
                S_set.add(x)
                reachable = reachable | (reachable << x)
                improved = True
                break # Restart scan to update missing list and reachable
                
        return list(S_set)

    best_set = []
    best_len = 0
    
    # Run many random restarts
    # With bitmask, each run is very fast.
    # Estimate: 100 items * few ops. ~1us?
    # 1s can do ~100k runs? Let's say 10k-100k runs.
    
    # Initial static orders
    static_orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 49, -1)) # Mix
    ]
    
    for o in static_orders:
        if time.time() > deadline - 5:
            break
        S = greedy(o)
        S = local_improve_add_only(S)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
        if best_len == 100:
            return best_set

    # Randomized restarts
    # We can also try "priority" based ordering, but pure random with many iterations is often robust.
    # Let's mix: 50% random, 50% sorted by "safety" (e.g. value > 100 are safest? No, all <= 100).
    # Elements > TARGET/2 (i.e. > 100) don't exist.
    # Elements in [101, 199] don't exist.
    # Elements in [1, 100].
    # Pairs summing to 200: (1,199)...(100,100).
    # Since max is 100, only 100+100 is a pair? No, 100 is unique.
    # So 100 cannot be part of a sum 200 with another 100.
    # But 100 + 99 + 1 = 200.
    
    while time.time() < deadline - 2:
        # Generate random order
        order = ALL[:]
        random.shuffle(order)
        
        # Sometimes bias towards larger numbers first (safer)
        if random.random() < 0.3:
            order.sort(reverse=True)
        
        S = greedy(order)
        S = local_improve_add_only(S)
        
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
            if best_len == 100:
                return best_set

    # Final polish: try to add any remaining
    if best_set:
        best_set = local_improve_add_only(best_set)

    if not best_set:
        return [100]
        
    return best_set
# EVOLVE-BLOCK-END
