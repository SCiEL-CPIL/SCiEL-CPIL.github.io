N = 18

# EVOLVE-BLOCK-START
def construct():
    def generate_sort_network(wires):
        n = len(wires)
        if n <= 1:
            return []
        if n == 2:
            return [(wires[0], wires[1])]
        
        k = n // 2
        left = wires[:k]
        right = wires[k:]
        
        result = []
        result.extend(generate_sort_network(left))
        result.extend(generate_sort_network(right))
        
        # Pre-merge
        for i in range(k):
            result.append((left[i], right[i]))
        
        # Odd-even merge
        result.extend(odd_even_merge(left, right))
        
        return result
    
    def odd_even_merge(A, B):
        m = len(A)
        if m == 0:
            return []
        if m == 1:
            return [(A[0], B[0])]
        
        A_odd = A[0::2]
        A_even = A[1::2]
        B_odd = B[0::2]
        B_even = B[1::2]
        
        result = []
        result.extend(odd_even_merge(A_odd, B_even))
        result.extend(odd_even_merge(A_even, B_odd))
        
        for i in range(m // 2):
            result.append((A[2*i], A[2*i+1]))
        
        return result
    
    comps = generate_sort_network(list(range(N)))
    # Remove duplicates while preserving order
    seen = set()
    unique_comps = []
    for c in comps:
        if c not in seen:
            seen.add(c)
            unique_comps.append(c)
    return unique_comps
# EVOLVE-BLOCK-END
