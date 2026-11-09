    def _score_factors(factors):
        if not factors:
            return 0
        n = 1
        for p, e in factors.items():
            n *= p ** e
        if n < 2:
            return 0
            
        # Generate divisors
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            p_powers = [1]
            for _ in range(e):
                p_powers.append(p_powers[-1] * p)
            for d in divs:
                for pp in p_powers:
                    new_divs.append(d * pp)
            divs = new_divs
            
        divs.sort()
        
        score = 0
        for i in range(1, len(divs)):
            if math.gcd(divs[i-1], divs[i]) == 1:
                score += 1
        return score
