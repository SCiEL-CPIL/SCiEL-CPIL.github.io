    if not fast_verify(best_net):
        # Fallback to odd-even if bitonic has a bug
        best_net = odd_even_network()
