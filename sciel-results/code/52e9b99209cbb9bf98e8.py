import math

def priority(vector, n):
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Smooth balance metric: log of product of (count + 1)
    # This rewards having all symbols present and balanced.
    # It is smoother than min() and more sensitive to imbalance than just sum of squares?
    # Actually, sum of squares is also smooth.
    # Let's use the log-entropy style score.
    balance_score = math.log(c0 + 1) + math.log(c1 + 1) + math.log(c2 + 1)
    
    s = sum(vector) % 3
    
    # Tie-breaker hash
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v) & 0x7FFFFFFF
    h_norm = h / 0x7FFFFFFF
    
    # Sum bonus: Encourage mixing residues?
    # Current: 1->2.0, 2->1.0, 0->0.0 with weight 0.01.
    # Let's try a slightly stronger effect but still secondary.
    sum_bonus = {1: 1.0, 2: 0.5, 0: 0.0}[s]
    
    # Return higher priority for better balance and preferred sum
    # Scale balance_score to be the main driver.
    # Max balance_score for n=10 is ~4.4.
    # Sum bonus max 1.0.
    # If we use weight 0.1 for sum, it's 0.1.
    # So balance is ~40x more important.
    
    return balance_score + sum_bonus * 0.1 + 1e-9 * h_norm
