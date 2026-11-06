import zlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Prioritizes vectors by their Hamming weight (number of non-zero coordinates).
    Vectors with lower weight are prioritized first.
    This structural ordering exploits the geometry of the vector space,
    potentially allowing the greedy algorithm to fill the space more efficiently
    by handling 'dense' regions (near the origin) before 'sparse' regions.
    """
    # Calculate the Hamming weight (count of non-zero elements)
    weight = sum(1 for x in vector if x != 0)
    
    # Secondary sort: Lexicographic order to ensure determinism and break ties
    # We use the lexicographic tuple itself as a tie-breaker.
    # To make it a single float, we can combine weight and lex order.
    # However, the evaluator sorts by priority.
    # If we return -weight, all weight 0 come first, then weight 1, etc.
    # Ties are broken by the evaluator's stable sort? 
    # The problem statement says "added greedily in descending priority".
    # If priorities are equal, the order is undefined or stable?
    # Usually, we should provide unique priorities or handle ties.
    # Let's add a small epsilon based on lexicographic rank to break ties.
    
    # To keep it simple and fast:
    # Return -weight. 
    # If the sort is stable, the input order matters.
    # The input order is likely arbitrary or lex.
    # To be safe, let's make the priority unique.
    
    # Encode the vector into a single integer to get a unique tie-breaker.
    # Base-3 representation.
    val = 0
    for i, x in enumerate(vector):
        val += x * (3 ** (n - 1 - i))
        
    # Primary key: Weight (lower is better -> higher priority -> more negative)
    # Secondary key: Lexicographic value (lower is better -> higher priority -> more negative)
    # We want to sort descending by priority.
    # So we want lower weight to have higher priority.
    # Priority = - (weight * BigConstant + val)
    
    big_const = 3 ** n
    return -float(weight * big_const + val)
# EVOLVE-BLOCK-END
