<<<<<<< SEARCH
def _prune_network(net, deadline):
    """
    Repeatedly remove redundant comparators until none are left.
    """
    current_net = net[:]
    changed = True
    while changed and time.time() < deadline:
        new_net = _remove_redundant(current_net)
        if len(new_net) < len(current_net):
            current_net = new_net
            changed = True
        else:
            changed = False
            
    return current_net
=======
def _prune_network(net, deadline):
    """
    Repeatedly remove redundant comparators until none are left.
    Uses BDD-based removal for precision.
    """
    current_net = net[:]
    changed = True
    while changed and time.time() < deadline:
        # Try BDD-based removal first for precision
        new_net = _remove_redundant_bdd(current_net)
        if len(new_net) < len(current_net):
            current_net = new_net
            changed = True
        else:
            # Fall back to bit-slicing if BDD didn't help
            new_net = _remove_redundant(current_net)
            if len(new_net) < len(current_net):
                current_net = new_net
                changed = True
            else:
                changed = False
            
    return current_net
>>>>>>> REPLACE

