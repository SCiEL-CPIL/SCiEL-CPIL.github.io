import sys
from fractions import Fraction

def construct():
    """
    Returns a nonempty subset of [1, 24] that maximizes cardinality
    such that no reciprocal of a selected element equals the sum of
    reciprocals of a subset of the other selected elements.
    """
    
    def is_valid(S):
        """
        Checks if the set S is reciprocal-subsum-free.
        Uses pruned dynamic programming with Fractions for exact arithmetic.
        """
        S_list = list(S)
        for x in S_list:
            target = Fraction(1, x)
            others = [y for y in S_list if y != x]
            
            # Set of reachable sums using a subset of 'others'
            # Start with 0 (empty subset)
            reachable = {Fraction(0)}
            
            for y in others:
                inv_y = Fraction(1, y)
                new_sums = set()
                for s in reachable:
                    ns = s + inv_y
                    # Prune: sums exceeding the target cannot lead to the target
                    # since all terms are positive
                    if ns <= target:
                        new_sums.add(ns)
                
                if new_sums:
                    reachable.update(new_sums)
                
                # If the target is reached, the condition is violated
                if target in reachable:
                    return False
            
        return True

    # Try to find the largest K such that the set of the K largest numbers is valid.
    # The set of largest numbers has the smallest reciprocals, making it the least
    # likely to contain a reciprocal subsum.
    for K in range(24, 0, -1):
        candidate = list(range(25 - K, 25))
        if is_valid(candidate):
            return candidate
    
    # Fallback, though K=1 is always valid
    return [1]

if __name__ == "__main__":
    result = construct()
    print(result)
